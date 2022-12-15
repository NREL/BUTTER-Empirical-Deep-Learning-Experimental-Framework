from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class TaskResultRecord():
    experiment_parameters: Dict[str, Any] # uniquely identify an experiment
    experiment_data: Dict[str, Any] # other data common to all runs of this experiment
    run_data: Dict[str, Any] # data about this run not in experiment_*
    run_history: Dict[str, Iterable] # per-epoch measurement history of this run
