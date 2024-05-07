from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Any, Tuple, Dict, TypeVar

from dmp.layer.layer import Layer
from dmp.model.network_info import NetworkInfo

T = TypeVar("T")


@dataclass
class ModelSpec(ABC):
    input: Optional[Layer] = None  # set to None for runtime determination
    output: Optional[Layer] = None  # set to None for runtime determination

    @abstractmethod
    def make_network(self) -> NetworkInfo:
        pass
