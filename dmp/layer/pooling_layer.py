from typing import Callable, Dict, Optional, Tuple, Any, List, Sequence, TypeVar, Union
from dmp.layer.spatitial_layer import ASpatitialLayer
from dmp.layer.layer import Layer, LayerConstructor, network_module_types

T = TypeVar('T')


class APoolingLayer(ASpatitialLayer):

    @property
    def strides(self) -> Tuple:
        config = self.config
        strides = config.get('strides', None)
        if strides is not None:
            return strides
        return config['pool_size']

    @staticmethod
    def make(
        layer_factory: LayerConstructor[T],
        pool_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> T:
        return layer_factory(config, input, {
            'pool_size': pool_size,
            'strides': strides,
        })


class MaxPool(APoolingLayer):

    @staticmethod
    def make(
        pool_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'MaxPool':
        return APoolingLayer.make(
            MaxPool,
            pool_size,
            strides,
            config,
            input,
        )


network_module_types.append(MaxPool)


class AvgPool(APoolingLayer):

    @staticmethod
    def make(
        pool_size: Sequence[int],
        strides: Sequence[int],
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'AvgPool':
        return APoolingLayer.make(
            AvgPool,
            pool_size,
            strides,
            config,
            input,
        )


network_module_types.append(AvgPool)


class AGlobalPoolingLayer(ASpatitialLayer):
    pass


class GlobalAveragePooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalAveragePooling)


class GlobalMaxPooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalMaxPooling)
