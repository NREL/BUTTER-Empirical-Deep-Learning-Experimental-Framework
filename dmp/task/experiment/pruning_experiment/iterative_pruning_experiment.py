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

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_kwcfg
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
import dmp.keras_interface.access_model_parameters as access_model_parameters

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.context import Context


@dataclass
class IterativePruningExperiment(TrainingExperiment):
    num_pruning_iterations: int

    pre_prune_epochs: int
    pre_pruning_trigger: Optional[KerasConfig]

    pruning_method: PruningMethod
    pruning_trigger: Optional[KerasConfig]
    max_pruning_epochs: int

    rewind: bool

    @property
    def version(self) -> int:
        return 1

    def __call__(
        self,
        context: Context,
    ) -> ExperimentResultRecord:
        # http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf Algorithim 2

        if self.pruning_trigger is None:
            self.pruning_trigger = self.early_stopping

        # tensorflow.config.optimizer.set_jit(True)
        self._setup_environment()
        dataset, metrics = self._load_and_prepare_dataset()

        # 1: Create a network with randomly initialization W0 ∈ Rd.
        # 2: Initialize pruning mask to m = 1d.
        network = self._make_network(self.model)
        model = self._make_model_from_network(network, metrics)
        experiment_history = self._try_restore_checkpoint(
            context,
            model,
            self.record.resume_from,
        )

        # 3: Train W0 to Wk with noise u ∼ U: Wk = A 0→k (W0, u).
        num_free_parameters = model.network.num_free_parameters
        early_stopping = make_keras_instance(self.pre_pruning_trigger)

        self._fit_model(
            context,
            self.fit,
            dataset,
            model,
            [early_stopping],
            epochs=self.pre_prune_epochs,
            experiment_history=experiment_history,
        )

        # save weights at this point for rewinding
        restore_point = io.BytesIO()
        model_serialization.save_model(model, restore_point)
        model_serialization.save_model_data(self, model, f"test_base")

        # 4: for n ∈ {1, . . . , N} do
        for iteration_n in range(self.num_pruning_iterations):
            # 5: Train m ⊙ Wk to m ⊙ WT with noise u ′∼ U:WT = Ak→T(m ⊙ Wk, u′).

            early_stopping = make_keras_instance(self.pruning_trigger)
            self._fit_model(
                context,
                self.fit,
                dataset,
                model,
                [early_stopping],
                epochs=self.max_pruning_epochs,
                experiment_history=experiment_history,
            )

            model_serialization.save_model_data(
                self, model, f"test_{iteration_n}_unpruned"
            )

            # 6: Prune the lowest magnitude entries of WT that remain. Let m[i] = 0 if WT [i] is pruned.
            self.pruning_method.prune(
                model.network.structure,
                model.keras_network.layer_to_keras_map,
            )

            model_serialization.save_model_data(
                self, model, f"test_{iteration_n}_pruned"
            )

            model.keras_model.summary()
            if self.rewind:
                model_serialization.load_model(
                    model,
                    restore_point,
                    load_mask=False,
                    load_optimizer=True,
                )

        # 7: Return Wk, m
        return self._record_result(
            context,
            dataset,
            model.network,
            experiment_history,
        )
