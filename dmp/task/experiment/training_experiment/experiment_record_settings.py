from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID
from dmp.task.experiment.training_experiment.model_saving_config import (
    ModelSavingConfig,
)
from dmp.task.experiment.training_experiment.model_state_resume_config import (
    ModelStateResumeConfig,
)


@dataclass
class RunSpecificConfig:
    post_training_metrics: bool
    times: bool
    model: Optional[Any]
    metrics: Optional[Any]

    # checkpoint_interval: float = 30 * 60.0 # how often to checkpoint in seconds (if implemented)

    model_saving: Optional[ModelSavingConfig] = None
    resume_from: Optional[
        ModelStateResumeConfig
    ] = None  # resume this experiment from the supplied checkpoint

    id: Optional[UUID] = None
    root_run: Optional[UUID] = None
    parent_run: Optional[UUID] = None
    sequence_run: Optional[UUID] = None
