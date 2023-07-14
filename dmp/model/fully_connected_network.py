from dataclasses import dataclass, field
import math
from typing import Any, Callable, List, Tuple, Dict
from dmp.common import make_dispatcher
from dmp.layer.flatten import Flatten
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo
from dmp.model.model_util import (
    find_closest_network_to_target_size_float,
    find_closest_network_to_target_size_int,
)
from dmp.layer import *


@dataclass
class FullyConnectedNetwork(ModelSpec, LayerFactory):
    widths: List[int] = field(default_factory=lambda: [4096])
    residual_mode: str = "none"  # 'none' or 'full'
    flatten_input: bool = True
    inner: Dense = field(
        default_factory=lambda: Dense.make(4096)
    )  # config of all but output layer (no units here)
    depth: int = -1  # len(widths) set by __post_init__()
    width: int = -1  # max(widths) set by __post_init__()
    min_width: int = -1  # set by __post_init__()
    rectangular: bool = False  # set by __post_init__()

    def __post_init__(self):
        self.depth = len(self.widths)
        self.width = max(self.widths)
        self.min_width = min(self.widths)
        self.rectangular = self.width == self.min_width

    def make_network(self) -> NetworkInfo:
        return NetworkInfo(
            self.make_layer(
                {},  # type: ignore
                [self.input],  # type: ignore
            ),
            {"widths": self.widths},
        )

    def make_layer(
        self,
        config: "LayerConfig",
        inputs: Union["Layer", List["Layer"]],
    ) -> Layer:
        if isinstance(inputs, Layer):
            inputs = [inputs]

        parent = inputs[0]
        if self.flatten_input:
            parent = Flatten({}, parent)

        # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
        widths = self.widths
        residual_mode = self.residual_mode
        for depth, width in enumerate(widths):
            layer = None
            if depth == len(widths) - 1:
                output = self.inner if self.output is None else self.output
                layer = output.make_layer({}, [parent])  # type: ignore
                layer.insert_if_not_exists(config)
            else:
                layer = self.inner.make_layer(config, [parent])
            layer["units"] = width

            # Skip connections for residual modes
            if residual_mode == "none":
                pass
            elif residual_mode == "full":
                # If this isn't the first or last layer, and the previous layer is
                # of the same width insert a residual sum between layers
                # NB: Only works for rectangle
                if depth > 0 and depth < len(widths) - 1 and width == widths[depth - 1]:
                    layer = Add({}, [layer, parent])
            else:
                raise NotImplementedError(f'Unknown residual mode "{residual_mode}".')

            parent = layer
        return parent
