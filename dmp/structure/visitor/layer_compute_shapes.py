import math
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeAlias, TypeVar, Union
import tensorflow.keras as keras
from tensorflow import Tensor
from cnn.cell_structures import DenseConvolutionalLayer, SeparableConvolutionalLayer
from dmp.structure.layer import Layer
from dmp.structure.layer_visitor import LayerVisitor
from dmp.task.aspect_test.aspect_test_utils import make_from_typed_config


class LayerComputeShapesVisitor(LayerVisitor[Tuple]):

    def __init__(self, target: Layer) -> None:
        self._layer_shape_map: Dict[Layer, Optional[Tuple]] = {}
        self._do_visit(target)

    def __call__(self) -> Dict[Layer, Tuple]:
        return self._layer_shape_map  # type: ignore

    def _do_visit(self, target: Layer) -> None:
        if target in self._layer_shape_map:
            return

        self._layer_shape_map[target] = None  # placeholder

        for i in target.inputs:
            self._do_visit(i)

        layer_shape = self._visit(target)
        self._layer_shape_map[target] = layer_shape

    def _visit_Input(self, target: Layer) -> Tuple:
        return target.config['shape']

    def _visit_Dense(self, target: Layer) -> Tuple:
        return (target.config['units'], )

    def _visit_Add(self, target: Layer) -> Tuple:
        return self._get_shape(target.input)

    def _visit_Concatenate(self, target: Layer) -> Tuple:
        axis = target.config['axis']
        total = sum((self._get_shape(i)[axis] for i in target.inputs))
        input_shape = self._get_shape(target.input)
        return input_shape[:axis] + (total, ) + input_shape[axis + 1:]

    def _visit_DenseConvolutionalLayer(self, target: Layer) -> Tuple:
        config = target.config
        input_conv_shape = self._get_shape(target.input)
        output_conv_shape = target.on_padding(
            lambda: input_conv_shape,
            lambda: (max(0, input_dim - kernel_dim + 1)
                     for input_dim, kernel_dim in zip(
                         input_conv_shape,
                         config['kernel_size'],
                     )),
        )

        return (*output_conv_shape, config['filters'])

    def _visit_SeparableConvolutionalLayer(self, target: Layer) -> Tuple:
        return self._visit_DenseConvolutionalLayer(target)

    def _visit_MaxPool(self, target: Layer) -> Tuple:
        config = target.config
        stride = config['stride']
        dimension = len(stride)

        input_conv_shape = self._get_shape(target.input)[dimension:]
        output_conv_shape = input_conv_shape
        delta = target.on_padding(
            lambda: (1, ) * dimension,
            lambda: config['pool_size'],
        )

        output_conv_shape = (\
                int(math.floor((i - d) / s)) + 1
                for i, d, s in zip(input_conv_shape,
                delta,
                stride))

        return (
            *output_conv_shape,
            self._get_num_input_channels(target, dimension),
        )

    def _visit_GlobalAveragePooling(self, target: Layer) -> Tuple:
        config = target.config
        dimension = self._get_and_delete_dimension(target)
        num_channels = (self._get_num_input_channels(target, dimension), )

        if config['keepdims']:
            ones = (1, ) * dimension
            return target.on_data_format(lambda: ones + num_channels,
                                         lambda: num_channels + ones)
        return num_channels

    def _visit_IdentityOperation(self, target: Layer) -> Tuple:
        return self._get_shape(target.input)

    def _visit_ZeroizeOperation(self, target: Layer) -> Tuple:
        return self._get_shape(target.input)

    def _get_shape(self, target: Layer) -> Tuple:
        result = self._layer_shape_map.get(target, None)
        if result is None:
            raise ValueError(f'Can not determine layer shapes.')
        return result

    def _get_num_input_channels(self, target: Layer, dimension: int) -> int:
        return \
            sum((max(1, sum(self._get_shape(i)[dimension:]))
            for i in target.inputs))

    def _get_and_delete_dimension(self, target: Layer) -> int:
        config = target.config
        dimension = config['dimension']
        del config['dimension']
        return dimension
