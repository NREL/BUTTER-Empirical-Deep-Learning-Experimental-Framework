from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from uuid import UUID

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord


class ExperimentResultLogger(ABC):
    @abstractmethod
    def log(
        self,
        record: ExperimentResultRecord,  # record to log
    ) -> None:
        pass
