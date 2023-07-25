from dataclasses import dataclass, field
import io
from pprint import pprint
from typing import List, Optional, Any, Dict, Tuple

from jobqueue.job import Job
import numpy
from dmp import parquet_util
from dmp.common import KerasConfig
import dmp.keras_interface.model_serialization as model_serialization
from dmp.layer.layer import Layer

from dmp.keras_interface.keras_utils import make_keras_instance, keras_kwcfg

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.context import Context
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class IterativePruningExperiment(TrainingExperiment):
    max_epochs_per_pruning_iteration: int
    pruning_iterations: int
    pruning_method: PruningMethod
    rewind: TrainingExperimentCheckpoint  # run-specific id
    new_seed: bool
    train_before_prune: bool

    @property
    def version(self) -> int:
        return 1

    def __call__(
        self,
        context: Context,
        run: RunSpec,
    ) -> None:
        # http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf Algorithim 2

        if self.pruning_trigger is None:
            self.pruning_trigger = self.early_stopping

        self._setup_environment(run)
        dataset, metrics = self._load_and_prepare_dataset()

        # 1: Create a network with randomly initialization W0 ∈ Rd.
        # 2: Initialize pruning mask to m = 1d.
        network = self._make_network(self.model)
        model = self._make_model_from_network(network, metrics)
        experiment_history = self._try_restore_checkpoint(context, run, model)

        # 4: for n ∈ {1, . . . , N} do
        first_iteration = True
        while True:
            epoch = self.get_current_epoch(experiment_history)
            iteration = epoch.model_number
            if iteration >= self.pruning_iterations:
                break

            if (
                not (first_iteration and self.train_before_prune)
                and epoch.model_epoch > 0
            ):
                iteration += 1

                # 6: Prune the lowest magnitude entries of WT that remain. Let m[i] = 0 if WT [i] is pruned.
                self.pruning_method.prune(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                )

                # load rewind point
                self.rewind.resume(model)

            # 3: Train W0 to Wk with noise u ∼ U: Wk = A 0→k (W0, u).
            # 5: Train m ⊙ Wk to m ⊙ WT with noise u ′∼ U:WT = Ak→T(m ⊙ Wk, u′).
            self._fit_model(
                context,
                run,
                self.fit,
                dataset,
                model,
                [self._make_early_stopping_callback()],
                experiment_history=experiment_history,
                new_model_number=False,
                new_seed=(self.new_seed and first_iteration),
                epochs=self.fit["epochs"],
            )

            # 7: Return Wk, m
            # save weights at this point
            self._save_checkpoint(
                context,
                run,
                dataset,
                network,
                experiment_history,
                model,
            )

            first_iteration = False

        context.update_summary()
