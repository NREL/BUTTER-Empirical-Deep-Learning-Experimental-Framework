from dataclasses import dataclass, field
from typing import Any, List


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NetworkModule:
    label: int = 0
    inputs: List['NetworkModule'] = field(default_factory=list)
    shape: List[int] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    @property
    def size(self) -> int:
        acc = 1
        for dim in self.shape:
            acc *= dim
        return acc
