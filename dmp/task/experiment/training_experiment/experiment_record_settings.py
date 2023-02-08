
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExperimentRecordSettings():
    post_training_metrics: bool  # new default false
    times: bool
    model: Optional[Any]
    metrics: Optional[Any]