from __future__ import annotations

from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

from dmp.task.task_result import TaskResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.context import Context


class Task(ABC):
    @abstractmethod
    def __call__(
        self,
        context: Context,
    ) -> TaskResult:
        pass

    def summary(self) -> None:
        """
        Pretty-prints a description of this Task.
        """

        from dmp.marshaling import marshal
        from pprint import pprint

        pprint(marshal.marshal(self))
