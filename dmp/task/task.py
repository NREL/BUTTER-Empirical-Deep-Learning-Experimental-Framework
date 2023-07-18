from __future__ import annotations

from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from dmp.task.task_result import TaskResult

from dmp.worker_task_context import WorkerTaskContext


@dataclass
class Task(ABC):
    @abstractmethod
    def __call__(
        self,
        context: WorkerTaskContext,
    ) -> TaskResult:
        pass

    @property
    def version(self) -> int:
        return 1

    def summary(self) -> None:
        """
        Pretty-prints a description of this Task.
        """

        from dmp.marshaling import marshal
        from pprint import pprint

        pprint(marshal.marshal(self))
