from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from dataclasses import dataclass


# ParameterPrimitive = Union[None, bool, int, float, str]
# ParameterValue = Union[ParameterPrimitive, List["ParameterValue"]]
# ParameterDict = Dict[str, "Parameter"]
# Parameter = Union[ParameterValue, ParameterDict]
# FlatParameterDict = Dict[str, ParameterValue]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.context import Context
    import pandas
    from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
    from dmp.task.experiment.training_experiment.run_spec import RunSpec


@dataclass
class Experiment(ABC):
    data: dict  # extra tags related to this experiment, including batch

    @property
    def version(self) -> int:
        return 100

    @abstractmethod
    def __call__(
        self,
        context: Context,
        run: RunSpec,
    ) -> None:
        pass

    @abstractmethod
    def summarize(
        self,
        results: List[pandas.DataFrame],
    ) -> Optional[ExperimentSummaryRecord]:
        pass
