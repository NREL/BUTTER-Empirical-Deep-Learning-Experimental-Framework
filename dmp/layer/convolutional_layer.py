from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.layer import Layer, LayerFactory, network_module_types
from dmp.layer.spatitial_layer import ASpatitialLayer

T = TypeVar('T')


class AConvolutionalLayer(ASpatitialLayer):

    @staticmethod
    def make(
        layer_factory: LayerFactory[T],
        filters: int,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        inputs: Union['Layer', List['Layer']],
    ) -> T:
        return layer_factory(config, inputs, {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
        })


class DenseConv(AConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'DenseConv':
        return AConvolutionalLayer.make(DenseConv, filters, kernel_size,
                                        strides, config, input)


network_module_types.append(DenseConv)


class ProjectionOperation(AConvolutionalLayer):
    pass


network_module_types.append(ProjectionOperation)


class SeparableConv(AConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'SeparableConv':
        return AConvolutionalLayer.make(SeparableConv, filters, kernel_size,
                                        strides, config, input)


network_module_types.append(SeparableConv)
