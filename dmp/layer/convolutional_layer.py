from abc import ABC
from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.layer import Layer, LayerConstructor, network_module_types
from dmp.layer.spatitial_layer import ASpatitialLayer

T = TypeVar('T')

class ConvolutionalLayer(ASpatitialLayer, ABC):

    @staticmethod
    def make(
        layer_factory: LayerConstructor[T],
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

