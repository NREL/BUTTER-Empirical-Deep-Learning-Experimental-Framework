from dataclasses import dataclass

from dmp.structure.network_module import NetworkModule


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NDense(NetworkModule):
    activation: str = 'relu'
