from dataclasses import dataclass
from typing import Optional

from dmp.structure.network_module import NetworkModule

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NBasicCell(NetworkModule):
    activation: str = 'relu'
    channels: int = 16
    kernel_regularizer : Optional[dict] = None # TODO pass regularizers into cell generator function
    bias_regularizer : Optional[dict] = None
    activity_regularizer : Optional[dict] = None

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConvStem(NBasicCell):
    batch_norm: bool = False
    input_channels: int = 3

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NCell(NBasicCell):
    batch_norm: bool = False
    operations: list = None
    nodes: int = 2
    cell_type: str = 'graph'

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NDownsample(NBasicCell):
    pass

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NFinalClassifier(NetworkModule):
    classes: int = 10   
    activation: str = 'softmax'