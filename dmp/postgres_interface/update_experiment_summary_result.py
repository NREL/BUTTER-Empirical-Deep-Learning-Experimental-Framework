from dataclasses import dataclass

from dmp.task.task_result import TaskResult


@dataclass
class UpdateExperimentSummaryResult(TaskResult):
    num_experiments_updated: int
