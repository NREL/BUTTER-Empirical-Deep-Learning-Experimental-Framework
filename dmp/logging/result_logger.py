from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Union

from dmp.task.task import Task


class ResultLogger(ABC):
    '''
    + get experiment id, log new experiment if not exists (using properties)
    + write run data to one or more tables

    + properties
    + run data (possibly different or multiple tables)
    '''

    @abstractmethod
    def log(
        self,
        experiment_parameters: Dict,
        run_parameters: Dict,
        result : Dict,
    ) -> None:
        pass


