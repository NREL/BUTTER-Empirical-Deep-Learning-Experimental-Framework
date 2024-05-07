from __future__ import annotations

from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from typing import Any, Dict, List, Optional, Tuple, Type, Union


from dmp.task.run_result import RunResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.context import Context
    from dmp.run_entry import RunEntry
    from dmp.task.experiment.training_experiment.run_spec import RunConfig


class RunCommand(ABC):
    @abstractmethod
    def __call__(
        self,
        context: Context,
    ) -> RunResult:
        pass

    def summary(self) -> None:
        """
        Pretty-prints a description of this Task.
        """

        from dmp.marshaling import marshal
        from pprint import pprint

        pprint(marshal.marshal(self))
