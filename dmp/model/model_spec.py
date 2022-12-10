from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Any, Tuple, Dict, TypeVar

from dmp.layer.layer import Layer
from dmp.layer.layer_factory import LayerFactory
from dmp.model.network_info import NetworkInfo

T = TypeVar('T')


@dataclass
class ModelSpec(ABC):
    inputs: List[Layer]  # set to empty list for runtime determination
    outputs: List[Layer]  # set to empty list for runtime determination

    @abstractmethod
    def make_network(self) -> NetworkInfo:
        pass

    @property
    def input(self) -> Layer:
        return self._get_one(self.inputs, 'input')

    @property
    def output(self) -> Layer:
        return self._get_one(self.outputs, 'output')

    def _get_one(self, seq: Sequence[T], name: str) -> T:
        if len(seq) != 1:
            raise NotImplementedError(
                f'Expected 1 {name} for model, but found {len(self.outputs)}.')
        return seq[0]
