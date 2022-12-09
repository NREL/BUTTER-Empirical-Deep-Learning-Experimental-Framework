from copy import copy
from functools import singledispatchmethod
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union

from dmp.layer import *


class WidthScaler:

    def __init__(self, target: Layer, scale_factor: float) -> None:
        self._scale_factor: float = scale_factor
        self._layer_map: Dict[Layer, Layer] = {}
        self._output = self._scale_network(target)

    def __call__(self) -> Tuple[Layer, Dict[Layer, Layer]]:
        return self._output, self._layer_map

    def _scale_network(self, target: Layer) -> Layer:
        layer_map = self._layer_map
        if target in layer_map:
            return layer_map[target]

        scaled_layer = copy(target)
        layer_map[target] = scaled_layer
        scaled_layer.inputs = [
            self._scale_network(input) for input in target.inputs
        ]
        self._scale_layer(scaled_layer)
        return scaled_layer

    @singledispatchmethod
    def _scale_layer(self, target: Layer) -> None:
        pass

    @_scale_layer.register
    def _(self, target: Dense) -> None:
        target.config['units'] = round(target.config['units'] *
                                       self._scale_factor)

    @_scale_layer.register
    def _(self, target: ConvolutionalLayer) -> None:
        target.config['filters'] = round(target.config['filters'] *
                                         self._scale_factor)