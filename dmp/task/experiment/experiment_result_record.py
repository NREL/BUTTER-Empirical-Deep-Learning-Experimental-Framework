from dataclasses import dataclass
from typing import Any, Dict, Iterable

from dmp.task.task_result import TaskResult


@dataclass
class ExperimentResultRecord(TaskResult):
    experiment_attrs: Dict[str, Any] # uniquely identify an experiment
    run_data: Dict[str, Any] # data about this run that's not in experiment_*
    run_history: Dict[str, Iterable] # per-epoch measurement history of this run
