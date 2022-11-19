from dataclasses import dataclass, field
from typing import List, Sequence

from dmp.structure.network_module import NetworkModule


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NInput(NetworkModule):
    shape: List[int] = field(default_factory=list)
    
    @property
    def output_shape(self) -> Sequence[int]:
        return self.shape

    @property
    def dimension(self) -> int:
        return len(self.shape)
