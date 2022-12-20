from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import AConvolutionalLayer
from dmp.layer.layer import register_layer_type, LayerConfig, Layer, empty_config, empty_inputs


class SeparableConv(AConvolutionalLayer):

    @staticmethod
    def make(
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: LayerConfig = empty_config,
        input: List[Layer] = empty_inputs,
    ) -> 'SeparableConv':
        return AConvolutionalLayer.make(SeparableConv, filters, kernel_size,
                                        strides, config, input)


register_layer_type(SeparableConv)
