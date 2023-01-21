from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Union

from jobqueue.job import Job
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.worker import Worker


@dataclass
class Task(ABC):
    
    @abstractmethod
    def __call__(self, worker: Worker, job: Job) -> TaskResult:
        pass

    @property
    def version(self) -> int:
        return -1


