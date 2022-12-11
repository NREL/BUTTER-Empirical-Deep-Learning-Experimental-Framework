from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import AConvolutionalLayer
from dmp.layer.layer import network_module_types, Layer


class SeparableConv(AConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'SeparableConv':
        return AConvolutionalLayer.make(SeparableConv, filters, kernel_size,
                                        strides, config, input)


network_module_types.append(SeparableConv)
