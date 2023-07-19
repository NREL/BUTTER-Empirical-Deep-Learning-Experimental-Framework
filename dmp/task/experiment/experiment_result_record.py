from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas
from dmp.task.run import Run

from dmp.task.task_result import TaskResult


@dataclass
class ExperimentResultRecord(TaskResult):
    run_history: pandas.DataFrame  # per-epoch measurement history of this run
    run_extended_history: Optional[pandas.DataFrame]  # additional history columns
