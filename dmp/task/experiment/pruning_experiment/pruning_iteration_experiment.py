from dataclasses import dataclass, field
import dataclasses
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
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.lth.pruning_config import PruningConfig

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)
from dmp.task.experiment.training_experiment.epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.task import Task
from dmp.worker import Worker
from dmp.worker_task_context import WorkerTaskContext


@dataclass
class PruningIterationExperiment(TrainingExperiment):
    """ """

    prune: PruningConfig  # contains run-specific num_iterations...
    rewind: TrainingExperimentCheckpoint  # run-specific id
    # TODO: what is run vs experiment attributes here?

    @property
    def version(self) -> int:
        return super().version + 1

    def __call__(
        self,
        context: WorkerTaskContext,
        new_seed: bool = False,
    ) -> ExperimentResultRecord:
        # NB: must have a compatible save mode
        self.record.save_trained_model = True  # type: ignore

        # tensorflow.config.optimizer.set_jit(True)
        self._set_random_seeds()
        dataset, metrics = self._load_and_prepare_dataset()
        network = self._make_network(self.model)
        model = self._make_model_from_network(network, metrics)

        # load pruning weights
        experiment_history = self._restore_checkpoint(
            context, model, self.record.resume_from
        )

        # prune network
        self.prune.method.prune(
            model.network.structure,
            model.keras_network.layer_to_keras_map,
        )

        # load rewind point
        self.rewind.resume(model)

        print(model.network.structure.summary())
        model.keras_model.summary()

        # train the model
        self._fit_model(
            context,
            self.fit,
            dataset,
            model,
            [self._make_early_stopping_callback()],
            experiment_history=experiment_history,
            new_seed=new_seed,
        )

        # enqueue next pruning iteration
        if self.prune.iteration < self.prune.num_iterations:
            child_task = PruningIterationExperiment(
                **vars(self),
            )

            child_task.record = dataclass.replace(
                child_task.record,
                resume_from=TrainingExperimentCheckpoint(
                    run_id=context.id,
                    load_mask=True,
                    load_optimizer=False,
                    epoch=self.get_current_epoch(experiment_history),
                ),
                parent_run=context.id,
            )
            context.push_task(child_task)

        return self._make_result_record(
            context,
            dataset,
            model.network,
            experiment_history,
        )
