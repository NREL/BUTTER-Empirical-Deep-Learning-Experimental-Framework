from dataclasses import dataclass

from dmp.structure.network_module import NetworkModule


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NAdd(NetworkModule):
    pass

    # def __init__(self,
    #              inputs: ('NetworkModule', ...) = (),
    #              ) -> None:
    #     super().__init__(inputs, inputs[0].shape)
