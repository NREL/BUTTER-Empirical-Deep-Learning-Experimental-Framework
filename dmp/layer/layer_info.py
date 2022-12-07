from dataclasses import dataclass
from typing import Tuple

from dmp.layer.layer import Layer


@dataclass
class LayerInfo():
    layer: Layer
    shape: Tuple[int, ...]
    num_parameters: int
