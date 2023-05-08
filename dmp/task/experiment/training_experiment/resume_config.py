from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from dmp.model.model_info import ModelInfo


@dataclass
class ResumeConfig(ABC):
    @abstractmethod
    def resume(
        self,
        model: ModelInfo,
    ) -> None:
        pass

    @abstractmethod
    def get_epoch(self) -> Tuple[int, int, int]:
        pass
