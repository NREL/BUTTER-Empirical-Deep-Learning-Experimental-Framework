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

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.model_state_resume_config import ModelStateResumeConfig
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.task import Task
from dmp.worker import Worker


@dataclass
class PruningIterationExperiment(TrainingExperiment):
    """
    """
    
    pruning_method: PruningMethod
    pruning_iteration: int
    pruning_max_iteration: int  # run-specific attribute

    # TODO: what is run vs experiment attributes here?
    rewind_point: ModelStateResumeConfig # run-specific id

    @property
    def version(self) -> int:
        return 1

    def __call__(
        self,
        worker: Worker,
        job: Job,
        *args,
        **kwargs,
    ) -> ExperimentResultRecord:
        with worker.strategy.scope():
            # tensorflow.config.optimizer.set_jit(True)
            self._set_random_seeds()
            dataset, metrics = self._load_and_prepare_dataset()
            network = self._make_network(self.model)
            model = self._make_model_from_network(network, metrics)
            self._resume_model(model, self.record.resume_from)
            print(model.network.structure.summary())
            model.keras_model.summary()

            history = self._fit_model(
                worker,
                job,
                self.fit,
                dataset,
                model,
                [self._make_early_stopping_callback()],
            )
            return self._make_result_record(
                worker.worker_info,
                job.id,
                dataset,
                model.network,
                history,
            )
            # TODO: if desired, enqueue next pruning/training iteration

            # # load prune source
            # # do pruning to get new mask
            # num_pruned = self.pruning_method.prune(
            #     model.network.structure, model.keras_network.layer_to_keras_map)
            # num_free_parameters = model.network.num_free_parameters - num_pruned

            # set weights from reset source
            # self._resume_model(model, self.rewind_point)

