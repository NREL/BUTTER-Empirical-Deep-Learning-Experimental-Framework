from dataclasses import dataclass
from typing import Optional

from dmp.structure.network_module import NetworkModule


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NDense(NetworkModule):
    activation: str = 'relu'
    kernel_regularizer : Optional[dict] = None
    bias_regularizer : Optional[dict] = None
    activity_regularizer : Optional[dict] = None
    kernel_initializer: Optional[dict] = 'glorot_uniform'
