from dataclasses import dataclass, field
import math
from typing import Any, Callable, List, Tuple, Dict
from dmp.common import make_dispatcher
from dmp.layer.flatten import Flatten
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo
from dmp.model.model_util import (
    find_closest_network_to_target_size_float,
    find_closest_network_to_target_size_int,
)
from dmp.layer import *


@dataclass
class DenseBySize(ModelSpec):
    shape: str = 'rectangle'  # (migrate) convert 'wide_first' to 'wide_first_10x'
    size: int = 4096  # (migrate)
    depth: int = 2  # (migrate)
    # input_layer : dict
    # input_layer.activation (migrate from input_activation)
    search_method: str = 'float'  # use 'integer' or 'float' network width search method
    inner: Dense = field(
        default_factory=lambda: Dense.make(16)
    )  # config of all but output layer (no units here)
    '''
        + activation (migrate)
        + kernel_regularizer (migrate)
        + bias_regularizer (migrate)
        + activity_regularizer (migrate)
        + batch_norm (new?)
    '''

    # output: dict  # output layer config (include units here)
    # residual: str  # (migrate from shape)

    # activation (migrate from runtime compute of output_activation)

    # @property
    # def num_outputs(self) -> int:
    #     return self.output['units']

    def make_network(self) -> NetworkInfo:
        shape = self.shape
        if isinstance(self.input, Input) and len(self.input['shape']) > 1:
            self.input = Flatten({}, [self.input])

        # TODO: make it so we don't need this hack?
        residual_mode = 'none'
        residual_suffix = '_residual'
        if shape.endswith(residual_suffix):
            residual_mode = 'full'
            shape = shape[0 : -len(residual_suffix)]

        widths_factory = _get_widths_factory(shape)

        if type(self.output) is not Dense:
            raise NotImplementedError('Invalid output type for model.')

        num_outputs = self.output['units']

        def make_network_with_scale(scale):
            widths = widths_factory(self, num_outputs, scale)

            return FullyConnectedNetwork(
                self.input,  # type: ignore
                self.output,  # type: ignore
                widths,
                residual_mode,
                False,
                self.inner,
            ).make_network()

        search_func = find_closest_network_to_target_size_int
        if self.search_method == 'integer':
            search_func = find_closest_network_to_target_size_int
        elif self.search_method == 'float':
            search_func = find_closest_network_to_target_size_float
        else:
            raise NotImplementedError(f'Unknown search method {self.search_method}.')

        delta, network = search_func(self.size, make_network_with_scale)

        # reject non-conformant network sizes
        delta = network.num_free_parameters - self.size
        relative_error = delta / self.size
        if abs(relative_error) >= 0.5:
            raise ValueError(
                f'Could not find conformant network error : {100 * relative_error}%, delta : {delta}, size: {self.size}, actual: {network.num_free_parameters}.'
            )

        return network


def _get_rectangular_widths(
    model: DenseBySize, num_outputs: int, scale: float
) -> List[int]:
    return ([round(scale)] * (model.depth - 1)) + [num_outputs]


def _get_trapezoidal_widths(
    model: DenseBySize, num_outputs: int, scale: float
) -> List[int]:
    beta = (scale - num_outputs) / (model.depth - 1)
    return [round(scale - beta * k) for k in range(0, model.depth)]


def _get_exponential_widths(
    model: DenseBySize, num_outputs: int, scale: float
) -> List[int]:
    beta = math.exp(math.log(num_outputs / scale) / (model.depth - 1))
    return [max(num_outputs, round(scale * (beta**k))) for k in range(0, model.depth)]


def _make_wide_first(
    first_layer_width_multiplier: float,
) -> Callable[[DenseBySize, int, float], List[int]]:
    def make_layout(model: DenseBySize, num_outputs: int, scale: float):
        depth = model.depth
        layout = []
        if depth > 1:
            layout.append(scale)
        if depth > 2:
            inner_width = max(1, int(round(scale / first_layer_width_multiplier)))
            layout.extend((inner_width for k in range(0, depth - 2)))
        layout.append(num_outputs)
        return layout

    return make_layout


_get_widths_factory = make_dispatcher(
    'shape',
    {
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
    },
)
