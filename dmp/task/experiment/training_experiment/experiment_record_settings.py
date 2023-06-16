
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID
from dmp.task.experiment.training_experiment.model_state_resume_config import ModelStateResumeConfig

from dmp.task.experiment.training_experiment.save_mode import SaveMode


@dataclass
class RunSpecificConfig():
    post_training_metrics: bool 
    times: bool
    model: Optional[Any]
    metrics: Optional[Any]
    model_saving: Optional[SaveMode] = None
    resume_from: Optional[ModelStateResumeConfig] = None # resume this experiment from the supplied checkpoint
    
    root_run : Optional[UUID] = None
    parent_run : Optional[UUID] = None
    sequence_run : Optional[UUID] = None