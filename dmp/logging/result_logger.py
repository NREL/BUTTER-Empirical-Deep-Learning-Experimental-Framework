from abc import ABC, abstractmethod
from typing import Dict
from uuid import UUID


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
        run_id: UUID,
        experiment_parameters: Dict,
        run_parameters: Dict,
        run_values: Dict,
        result: Dict,
    ) -> None:
        pass
