from dataclasses import dataclass
from typing import Dict, Optional
from uuid import UUID

from traitlets import Any

from dmp.task.experiment.model_saving.model_saving_spec import (
    ModelSavingSpec,
)
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class RunSpec:
    seed: int

    data: dict  # run-specific information, e.g. runtime environment

    record_post_training_metrics: bool
    record_times: bool

    model_saving: Optional[ModelSavingSpec]

    # point to resume from (None means to start fresh)
    resume_checkpoint: Optional[TrainingExperimentCheckpoint]

    # checkpoint_interval: float = 30 * 60.0 # how often to checkpoint in seconds (if implemented)
