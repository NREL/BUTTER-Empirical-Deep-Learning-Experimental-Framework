
from dataclasses import dataclass
from dmp.task.task import Task


@dataclass
class ExperimentSummaryUpdater(Task):
    def __call__(self, worker: Worker, job: Job) -> TaskResultRecord:
        pass