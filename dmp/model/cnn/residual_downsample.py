from dataclasses import dataclass
from typing import List, Sequence

from dmp.layer import *

@dataclass
class ResidualDownsample(LayerFactory):
    stride: List[int]
    pooling: LayerFactory

    def make_layer(self, inputs: List[Layer]) -> Layer:
        return inputs[0]