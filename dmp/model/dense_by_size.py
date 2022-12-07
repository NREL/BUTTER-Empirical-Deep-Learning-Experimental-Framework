from dataclasses import dataclass
import math
from typing import Any, Tuple, Dict
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo
from dmp.task.task_util import find_best_layout_for_budget_and_depth, make_dispatcher
from dmp.layer import *


@dataclass
class DenseBySize(ModelSpec):
    shape: str  # (migrate) convert 'wide_first' to 'wide_first_10x'
    size: int  # (migrate)
    depth: int  # (migrate)
    # input_layer : dict
    #input_layer.activation (migrate from input_activation)
    layer: dict  # config of all but output layer (no units here)
    '''
        + activation (migrate)
        + kernel_regularizer (migrate)
        + bias_regularizer (migrate)
        + activity_regularizer (migrate)
        + batch_norm (new?)
    '''
    output: dict  # output layer config (include units here)
    # residual: str  # (migrate from shape)

    # activation (migrate from runtime compute of output_activation)

    @property
    def num_outputs(self) -> int:
        return self.output['units']

    def make_network(self) -> NetworkInfo:
        shape = self.shape

        #TODO: make it so we don't need this hack?
        residual_mode = 'none'
        residual_suffix = '_residual'
        if shape.endswith(residual_suffix):
            residual_mode = 'full'
            shape = shape[0:-len(residual_suffix)]

        widths_factory = _widths_factory(shape)

        def make_network_with_scale(scale):
            widths = widths_factory(self, scale)
            return NetworkInfo(
                self._make_network_from_widths(residual_mode, widths),
                {'widths': widths},
            )

        delta, network = find_best_layout_for_budget_and_depth(
            self.size,
            make_network_with_scale,
        )

        # reject non-conformant network sizes
        delta = network.num_free_parameters - self.size
        relative_error = delta / self.size
        if abs(relative_error) > .2:
            raise ValueError(
                f'Could not find conformant network error : {relative_error}%, delta : {delta}, size: {self.size}.'
            )

        return network

    # TODO: wrap make_network_from_widths up with shape to width function and have it call find_best_layout_for_budget_and_depth above

    def _make_network_from_widths(
        self,
        residual_mode: str,
        widths: List[int],
    ) -> Layer:
        parent = Input({'shape': self.input_shape}, [])
        # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
        for depth, width in enumerate(widths):
            if depth == len(widths) - 1:
                layer_config = self.output
            else:
                layer_config = self.layer
            layer_config = layer_config.copy()

            # Fully connected layer
            layer_config['units'] = width
            current_layer = Dense(layer_config, parent)

            # Skip connections for residual modes
            if residual_mode == 'none':
                pass
            elif residual_mode == 'full':
                # If this isn't the first or last layer, and the previous layer is
                # of the same width insert a residual sum between layers
                # NB: Only works for rectangle
                if depth > 0 and depth < len(widths) - 1 and \
                    width == widths[depth - 1]:
                    current_layer = Add({}, [current_layer, parent])
            else:
                raise NotImplementedError(
                    f'Unknown residual mode "{residual_mode}".')

            parent = current_layer
        return parent


def _get_rectangular_widths(model: DenseBySize, scale: float) -> List[int]:
    return (([round(scale)] * (model.depth - 1)) + [model.num_outputs])


def _get_trapezoidal_widths(model: DenseBySize, scale: float) -> List[int]:
    beta = (scale - model.num_outputs) / (model.depth - 1)
    return [round(scale - beta * k) for k in range(0, model.depth)]


def _get_exponential_widths(model: DenseBySize, scale: float) -> List[int]:
    beta = math.exp(math.log(model.num_outputs / scale) / (model.depth - 1))
    return [
        max(model.num_outputs, round(scale * (beta**k)))
        for k in range(0, model.depth)
    ]


def _make_wide_first(
    first_layer_width_multiplier: float,
) -> Callable[[DenseBySize, float], List[int]]:

    def make_layout(model: DenseBySize, scale: float):
        depth = model.depth
        layout = []
        if depth > 1:
            layout.append(scale)
        if depth > 2:
            inner_width = max(1,
                              int(round(scale / first_layer_width_multiplier)))
            layout.extend((inner_width for k in range(0, depth - 2)))
        layout.append(model.num_outputs)
        return layout

    return make_layout


_widths_factory = make_dispatcher(
    'shape', {
        'rectangle': _get_rectangular_widths,
        'trapezoid': _get_trapezoidal_widths,
        'exponential': _get_exponential_widths,
        'wide_first_2x': _make_wide_first(2),
        'wide_first_4x': _make_wide_first(4),
        'wide_first_5x': _make_wide_first(5),
        'wide_first_8x': _make_wide_first(8),
        'wide_first_10x': _make_wide_first(10),
        'wide_first_16x': _make_wide_first(16),
        'wide_first_20x': _make_wide_first(20),
    })
