from dataclasses import dataclass
from typing import Sequence


@dataclass
class NetworkSpecification():
    input_shape: Sequence[int]
    # output_shape: Sequence[int]
