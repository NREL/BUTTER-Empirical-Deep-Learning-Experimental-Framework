
from dataclasses import dataclass
from typing import Any, Optional
from dmp.task.experiment.training_experiment.resume_config import ResumeConfig

from dmp.task.experiment.training_experiment.save_mode import SaveMode


@dataclass
class ExperimentRecordSettings():
    post_training_metrics: bool 
    times: bool
    model: Optional[Any]
    metrics: Optional[Any]
    model_saving: Optional[SaveMode] = None
    resume_from: Optional[ResumeConfig] = None # resume this experiment from the supplied checkpoint