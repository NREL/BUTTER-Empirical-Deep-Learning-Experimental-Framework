from typing import Callable, Dict, Optional, Tuple, Any, List, Sequence, TypeVar, Union
from dmp.layer.layer import register_layer_type
from dmp.layer.pooling_layer import APoolingLayer
from dmp.layer.spatitial_layer import ASpatitialLayer



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


register_layer_type(AvgPool)