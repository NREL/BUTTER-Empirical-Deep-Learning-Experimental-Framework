from functools import singledispatchmethod
import math
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeAlias, TypeVar, Union
from dmp.structure.layer import *


class ComputeFreeParametersVisitor:

    def __init__(
        self,
        target: Layer,
        layer_shapes: Dict[Layer, Tuple],
    ) -> None:
        self._layer_shapes: Dict[Layer, Tuple] = layer_shapes
        self._layer_parameters: Dict[Layer, int] = \
            {layer : self._visit(layer) for layer in target.all_descendants}
        self._num_free_parameters: int = \
            sum((n for n in self._layer_parameters.values()))

    def __call__(self) -> Tuple[int, Dict[Layer, int]]:
        return self._num_free_parameters, self._layer_parameters

    def _get_shape(self, target: Layer) -> Tuple:
        return self._layer_shapes[target]

    def _get_size(self, target: Layer) -> int:
        return sum(self._get_shape(target))

    @singledispatchmethod
    def _visit(self, target: Layer, config: Dict) -> int:
        return 0

    @_visit.register
    def _(self, target: Dense, config: Dict) -> int:
        return config['units'] * \
            (sum((self._get_size(i) for i in target.inputs)) +\
                 (1 if target.use_bias else 0))

    @_visit.register
    def _(self, target: AConvolutionalLayer, config: Dict) -> int:
        return self._get_count_for_conv_layer(
            target,
            config,
            math.prod(config['kernel_size']),
        )

    @_visit.register
    def _(self, target: SeparableConvolutionalLayer, config: Dict) -> int:
        return self._get_count_for_conv_layer(
            target,
            config,
            sum(config['kernel_size']),
        )

    def _get_count_for_conv_layer(self, target, config, num_nodes_per_filter):
        num_nodes = num_nodes_per_filter * config['filters']

        input_conv_shape, input_channels = \
            target.to_conv_shape_and_channels(self._get_shape(target.input))

        params_per_node = input_channels + (1 if target.use_bias else 0)
        return num_nodes * params_per_node


def compute_free_parameters(
    target: Layer,
    layer_shapes: Dict[Layer, Tuple],
) -> Tuple[int, Dict[Layer, int]]:
    return ComputeFreeParametersVisitor(target, layer_shapes)()