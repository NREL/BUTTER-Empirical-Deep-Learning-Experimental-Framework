from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
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
        results: List[Tuple[UUID, UUID, Dict]],
    ) -> None:
        pass
