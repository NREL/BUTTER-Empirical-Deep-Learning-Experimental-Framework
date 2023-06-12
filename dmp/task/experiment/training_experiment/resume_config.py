from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from dmp.model.model_info import ModelInfo
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


@dataclass
class ResumeConfig(ABC):
    @abstractmethod
    def resume(
        self,
        model: ModelInfo,
    ) -> None:
        pass

    @abstractmethod
    def get_epoch(self) -> TrainingEpoch:
        pass
