from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import Layer, network_module_types


class DenseConv(ConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'DenseConv':
        return ConvolutionalLayer.make(DenseConv, filters, kernel_size,
                                       strides, config, input)

    @staticmethod
    def makeNxN(n: int,
                input: Union['Layer', List['Layer']] = []) -> 'DenseConv':
        return DenseConv.make(-1, [n, n], [1, 1], {}, input)

def conv1x1(input: Union['Layer', List['Layer']] = []) -> DenseConv:
    return DenseConv.makeNxN(1, input)

def conv3x3(input: Union['Layer', List['Layer']] = []) -> DenseConv:
    return DenseConv.makeNxN(3, input)

def conv5x5(input: Union['Layer', List['Layer']] = []) -> DenseConv:
    return DenseConv.makeNxN(5, input)


network_module_types.append(DenseConv)
