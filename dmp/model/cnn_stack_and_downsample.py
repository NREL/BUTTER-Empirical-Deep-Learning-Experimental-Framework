from dataclasses import dataclass
from typing import List
from dmp.task.model_spec.model_spec import ModelSpec

@dataclass
class CNNStackAndDownsample(ModelSpec):
    num_stacks: int
    cells_per_stack: int
    stem: dict
    cell_operations: List[List[str]]  #(and/or preset operations name?)
    cell_conv: dict
    cell_pooling: dict
    downsample_conv: dict
    downsample_pooling: dict
    output: dict
