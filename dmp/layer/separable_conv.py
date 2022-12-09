from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import network_module_types, Layer


class SeparableConv(ConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'SeparableConv':
        return ConvolutionalLayer.make(SeparableConv, filters, kernel_size,
                                        strides, config, input)


network_module_types.append(SeparableConv)
