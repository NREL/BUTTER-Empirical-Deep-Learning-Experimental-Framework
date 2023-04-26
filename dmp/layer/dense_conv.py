from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import Layer, empty_config, empty_inputs, LayerConfig

class DenseConv(ConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: LayerConfig = empty_config,
        inputs: Union['Layer', List['Layer']] = empty_inputs,
    ) -> 'DenseConv':
        return ConvolutionalLayer.make(DenseConv, filters, kernel_size,
                                       strides, config, inputs)

    @staticmethod
    def make_NxN(n: int,
                inputs: List[Layer] = empty_inputs,
                config: LayerConfig = empty_config,
                ) -> 'DenseConv':
        return DenseConv.make(-1, [n, n], [1, 1], config, inputs)

def conv_1x1(**kwargs) -> DenseConv:
    return DenseConv.make_NxN(1, **kwargs)

def conv_3x3(**kwargs) -> DenseConv:
    return DenseConv.make_NxN(3, **kwargs)

def conv_5x5(**kwargs) -> DenseConv:
    return DenseConv.make_NxN(5, **kwargs)


