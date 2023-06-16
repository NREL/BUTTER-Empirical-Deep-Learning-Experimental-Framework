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
from dmp.task.experiment.pruning_experiment.pruning_iteration_experiment import PruningIterationExperiment

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.model_state_resume_config import ModelStateResumeConfig
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.task import Task
from dmp.worker import Worker
from dmp.worker_task_context import WorkerTaskContext


@dataclass
class LTHSeedChangeExperiment(PruningIterationExperiment):
    """
    """

    @property
    def version(self) -> int:
        return super().version + 1

    def __call__(
        self,
        context: WorkerTaskContext,
    ) -> ExperimentResultRecord:
        result_record: ExperimentResultRecord = super(
            TrainingExperiment, self)(context)  # type: ignore

        # TODO: mark seed change epoch somehow

        # enqueue pruning iteration experiment
        child = PruningIterationExperiment(**vars(self))
        child.record = dataclass.replace(
            child.record,
            resume_from=ModelStateResumeConfig(
                run_id=context.id,
                load_mask=True,
                load_optimizer=False,
                epoch=TrainingEpoch(
                    # epoch=resume_p
                )
            ),
            parent_run=context.id,
        )
        context.push_task(child)

        return result_record
