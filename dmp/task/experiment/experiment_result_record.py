from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas

from dmp.task.task_result import TaskResult


@dataclass
class ExperimentResultRecord(TaskResult):
    experiment_attrs: Dict[str, Any] # uniquely identify an experiment
    experiment_properties: Dict[str, Any] # additional data related to this experiment
    run_data: Dict[str, Any] # data about this run that's not in experiment_*
    run_history: pandas.DataFrame # per-epoch measurement history of this run
    run_extended_history: Optional[pandas.DataFrame] # additional history columns
