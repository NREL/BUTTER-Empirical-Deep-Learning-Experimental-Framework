from typing import Callable, Dict, Optional, Tuple, Any, List, Sequence, TypeVar, Union
from dmp.layer.pooling_layer import APoolingLayer
from dmp.layer.spatitial_layer import ASpatitialLayer
from dmp.layer.layer import Layer, LayerConstructor, network_module_types, LayerConfig, empty_config, empty_inputs



class AvgPool(APoolingLayer):

    @staticmethod
    def make(
        pool_size: Sequence[int],
        strides: Sequence[int],
        *args,
        **kwargs,
    ) -> 'AvgPool':
        return APoolingLayer.make(
            AvgPool,
            pool_size,
            strides,
            *args,
            **kwargs,
        )


network_module_types.append(AvgPool)