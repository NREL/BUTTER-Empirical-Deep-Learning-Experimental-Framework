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
from dmp.task.experiment.lth.pruning_config import PruningConfig

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_run_spec import (
    IterativePruningRunSpec,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.context import Context
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class IterativePruningExperiment(TrainingExperiment):
    pruning: PruningConfig

    @property
    def version(self) -> int:
        return 1

    def __call__(
        self,
        context: Context,
        run: IterativePruningRunSpec,
    ) -> None:
        print(f"********** 1")
        # http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf Algorithim 2
        pruning = self.pruning

        self._setup_environment(run)
        dataset, metrics = self._load_and_prepare_dataset()

        # 1: Create a network with randomly initialization W0 ∈ Rd.
        # 2: Initialize pruning mask to m = 1d.
        network = self._make_network(self.model)
        model = self._make_model_from_network(network, metrics)
        experiment_history = self._try_restore_checkpoint(context, run, model)

        rewind_point = TrainingExperimentCheckpoint(
            run_id=run.rewind_run_id,
            load_mask=False,
            load_optimizer=pruning.rewind_optimizer,
            epoch=pruning.rewind_epoch,
        )

        print(f"********** 2")

        # 4: for n ∈ {1, . . . , N} do
        first_iteration = True
        while True:
            epoch = self.get_current_epoch(experiment_history)
            iteration = epoch.model_number

            print(f"********** 3 {epoch}")
            if iteration >= pruning.iterations:
                print(f"********** 4")
                break

            print(f"********** 5")
            is_new_iteration = run.prune_first_iteration or not first_iteration
            if is_new_iteration:
                iteration += 1
                print(f"********** 6")

                # 6: Prune the lowest magnitude entries of WT that remain. Let m[i] = 0 if WT [i] is pruned.
                pruning.method.prune(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                )

                # load rewind point
                rewind_point.resume(model)
                print(f"********** 7")

            print(f"********** 8")
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
                new_model_number=is_new_iteration,
                new_seed=(pruning.new_seed and first_iteration),
                epochs=pruning.max_epochs_per_iteration,
            )

            print(f"********** 9")

            # 7: Return Wk, m
            # save weights at this point
            run.prune_first_iteration = True
            pruning.new_seed = False
            self._save_checkpoint(
                context,
                run,
                dataset,
                network,
                experiment_history,
                model,
            )
            print(f"********** 10")

            first_iteration = False

        print(f"********** 11")
        self._record_result(
            context,
            run,
            dataset,
            model.network,
            experiment_history,
        )

        print(f"********** 12")

        context.update_summary()
        print(f"********** 13")
