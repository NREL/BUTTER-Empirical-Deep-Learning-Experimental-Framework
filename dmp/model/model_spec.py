from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Any, Tuple, Dict

from dmp.layer.layer import Layer


@dataclass
class ModelSpec(ABC):
    input_shape: Sequence[int]
    # output_shape: Sequence[int]

    @abstractmethod
    def make_network(self) -> Tuple[Layer, int, Dict[Layer, Tuple]]:
        pass
