from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Union

from jobqueue.job import Job
from dmp.task.task_result import TaskResult
from dmp.worker import Worker


@dataclass
class Task(ABC):

    @abstractmethod
    def __call__(self, worker: Worker, job: Job,
                 *args,
                 **kwargs,
                 ) -> TaskResult:
        pass

    @property
    def version(self) -> int:
        return 0

    
    def summary(self) -> None:
        '''
        Pretty-prints a description of this Task.
        '''

        from dmp.marshaling import marshal
        from pprint import pprint

        pprint(marshal.marshal(self))
    
