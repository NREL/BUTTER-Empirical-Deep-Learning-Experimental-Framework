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
from dmp.task.experiment.experiment import ExperimentTask
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.pruning_experiment.pruning_iteration_experiment import PruningIterationExperiment

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import TrainingExperimentCheckpoint
from dmp.task.experiment.training_experiment.epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.task import Task
from dmp.worker import Worker
from dmp.context import Context


@dataclass
class LTHSeedChangeExperiment(PruningIterationExperiment):
    """

    """

    pruning: PruningConfig  # contains run-specific num_iterations...
    rewind: TrainingExperimentCheckpoint  # run-specific id
    # TODO: what is run vs experiment attributes here?

    @property
    def version(self) -> int:
        return super().version + 1

    def __call__(
        self,
        context: Context,
    ) -> ExperimentResultRecord:
        result_record: ExperimentResultRecord = super()(
            context,
            new_seed=True,
        )  # type: ignore

        # enqueue pruning iteration experiment
        child = PruningIterationExperiment(**vars(self))
        child.record = dataclass.replace(
            child.record,
            resume_from=TrainingExperimentCheckpoint(
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

    def _append_fit_history_to_model_history(
        self,
        new_model_number: bool,
        experiment_history: Optional[Dict[str, Any]],
        fit_history: Dict[str, Any],
    ) -> Dict[str, Any]:

        fit_history[self.keys.seed_number] =

        return super()._append_fit_history_to_model_history(
            new_model_number,
            experiment_history,
            fit_history,
        )
