from typing import Callable, Dict, Optional, Tuple, Any, List, Sequence, TypeVar, Union
from dmp.layer.layer import register_layer_type
from dmp.layer.pooling_layer import PoolingLayer
from dmp.layer.spatitial_layer import SpatitialLayer



class AvgPool(PoolingLayer):

    @staticmethod
    def make(
        pool_size: Sequence[int],
        strides: Sequence[int],
        *args,
        **kwargs,
    ) -> 'AvgPool':
        return PoolingLayer.make(
            AvgPool,
            pool_size,
            strides,
            *args,
            **kwargs,
        )


register_layer_type(AvgPool)