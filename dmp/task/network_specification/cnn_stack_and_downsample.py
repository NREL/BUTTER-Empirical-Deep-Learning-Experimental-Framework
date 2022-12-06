from dataclasses import dataclass
from typing import List
from dmp.task.network_specification.network_specification import NetworkSpecification

@dataclass
class CNNStackAndDownsample(NetworkSpecification):
    num_stacks: int
    cells_per_stack: int
    stem: dict
    cell_operations: List[List[str]]  #(and/or preset operations name?)
    cell_conv: dict
    cell_pooling: dict
    downsample_conv: dict
    downsample_pooling: dict
    output: dict
