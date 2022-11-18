from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from dmp.structure.network_module import NetworkModule


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NBasicCNN(NetworkModule):
    activation: str = 'relu'
    filters: int = 16
    kernel_size: int = 3
    stride: int = 1
    padding: str = 'same'
    kernel_regularizer: Optional[dict] = None
    bias_regularizer: Optional[dict] = None
    activity_regularizer: Optional[dict] = None

    @property
    def inputs(self) -> List['NBasicCNN']:
        return super().inputs  # type: ignore

    @property
    def dimension(self) -> int:
        return self.inputs[0].dimension


# @dataclass(frozen=False, eq=False, unsafe_hash=False)
# class NCNNInput(NetworkModule):
#     filters: int = 16


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConv(NBasicCNN):
    batch_norm: str = 'none'

    @property
    def num_free_parameters_in_module(self) -> int:
        params = (self.kernel_size**2) * self.filters
        for i in self.inputs:
            params *= i.filters
        return params


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NSepConv(NBasicCNN):
    batch_norm: str = 'none'

    @property
    def num_free_parameters_in_module(self) -> int:
        params = 2 * self.kernel_size * self.filters
        for i in self.inputs:
            params *= i.filters
        return params


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NMaxPool(NBasicCNN):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NGlobalPool(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NIdentity(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NZeroize(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConcat(NetworkModule):
    pass