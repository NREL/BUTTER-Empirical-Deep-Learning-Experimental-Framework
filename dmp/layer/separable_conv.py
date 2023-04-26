from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import LayerConfig, Layer, empty_config, empty_inputs


class SeparableConv(ConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: LayerConfig = empty_config,
        input: Union['Layer', List['Layer']] = empty_inputs,
    ) -> 'SeparableConv':
        return ConvolutionalLayer.make(SeparableConv, filters, kernel_size,
                                        strides, config, input)


