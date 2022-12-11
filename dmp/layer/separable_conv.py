from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import AConvolutionalLayer
from dmp.layer.layer import network_module_types, Layer, empty_inputs, empty_config


class SeparableConv(AConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: Dict[str, Any] = empty_config,
        input: Union['Layer', List['Layer']] = empty_inputs,
    ) -> 'SeparableConv':
        return AConvolutionalLayer.make(SeparableConv, filters, kernel_size,
                                        strides, config, input)


network_module_types.append(SeparableConv)
