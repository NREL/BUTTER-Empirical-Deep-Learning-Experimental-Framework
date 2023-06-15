
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID
from dmp.task.experiment.training_experiment.resume_config import ResumeConfig

from dmp.task.experiment.training_experiment.save_mode import SaveMode


@dataclass
class RunSpecificConfig():
    post_training_metrics: bool 
    times: bool
    model: Optional[Any]
    metrics: Optional[Any]
    model_saving: Optional[SaveMode] = None
    resume_from: Optional[ResumeConfig] = None # resume this experiment from the supplied checkpoint
    root_run : Optional[UUID] = None
    parent_run : Optional[UUID] = None