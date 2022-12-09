from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import Layer, network_module_types

class DenseConv(ConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'DenseConv':
        return ConvolutionalLayer.make(DenseConv, filters, kernel_size,
                                        strides, config, input)


network_module_types.append(DenseConv)

