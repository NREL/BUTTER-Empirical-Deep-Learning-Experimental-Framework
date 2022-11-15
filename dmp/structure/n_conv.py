from dataclasses import dataclass
from typing import Optional

from dmp.structure.network_module import NetworkModule


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NBasicCNN(NetworkModule):
    activation: str = 'relu'
    channels: int = 16
    kernel_size: int = 3
    stride: int = 1 
    padding: str = 'same'
    kernel_regularizer : Optional[dict] = None
    bias_regularizer : Optional[dict] = None
    activity_regularizer : Optional[dict] = None

# @dataclass(frozen=False, eq=False, unsafe_hash=False)
# class NCNNInput(NetworkModule):
#     channels: int = 16

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConv(NBasicCNN):
    batch_norm: str = 'none'
@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NSepConv(NBasicCNN):
    batch_norm: str = 'none'

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