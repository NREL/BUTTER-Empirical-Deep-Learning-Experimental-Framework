
from dataclasses import dataclass
from typing import Any, Optional

from dmp.task.experiment.training_experiment.save_mode import SaveMode


@dataclass
class ExperimentRecordSettings():
    post_training_metrics: bool 
    times: bool
    model: Optional[Any]
    metrics: Optional[Any]
    model_saving: Optional[SaveMode] = None