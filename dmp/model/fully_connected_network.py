from dataclasses import dataclass
import math
from typing import Any, Callable, List, Tuple, Dict
from dmp.common import make_dispatcher
from dmp.layer.flatten import Flatten
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo
from dmp.model.model_util import find_closest_network_to_target_size_float, find_closest_network_to_target_size_int
from dmp.layer import *


@dataclass
class FullyConnectedNetwork(ModelSpec, LayerFactory):
    widths: List[int]
    residual_mode: str
    flatten_input: bool
    inner: Dense  # config of all but output layer (no units here)

    def make_network(self) -> NetworkInfo:
        return NetworkInfo(
            self.make_layer(
                [self.input],  # type: ignore
                {},
            ),
            {'widths': self.widths},
        )

    def make_layer(
        self,
        inputs: List[Layer],
        config: 'LayerConfig',
    ) -> Layer:
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
                layer = output.make_layer([parent], {})  # type: ignore
                layer.insert_if_not_exists(config)
            else:
                layer = self.inner.make_layer([parent], config)
            layer['units'] = width

            # Skip connections for residual modes
            if residual_mode == 'none':
                pass
            elif residual_mode == 'full':
                # If this isn't the first or last layer, and the previous layer is
                # of the same width insert a residual sum between layers
                # NB: Only works for rectangle
                if depth > 0 and depth < len(widths) - 1 and \
                    width == widths[depth - 1]:
                    layer = Add({}, [layer, parent])
            else:
                raise NotImplementedError(
                    f'Unknown residual mode "{residual_mode}".')

            parent = layer
        return parent
