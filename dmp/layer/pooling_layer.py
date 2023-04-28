from abc import ABC
from typing import Callable, Dict, Optional, Tuple, Any, List, Sequence, TypeVar, Union
from dmp.layer.spatitial_layer import SpatitialLayer
from dmp.layer.layer import Layer, LayerConstructor, LayerConfig, empty_config, empty_inputs


T = TypeVar('T')


class PoolingLayer(SpatitialLayer, ABC):

    @property
    def strides(self) -> Tuple:
        config = self.config
        strides = config.get('strides', None)
        if strides is not None:
            return strides
        return config['pool_size']

    @classmethod
    def make(
        cls,
        layer_factory: LayerConstructor[T],
        pool_size: Sequence[int],
        strides: Sequence[int],
        config: LayerConfig = empty_config,
        input: Union['Layer', List['Layer']] = empty_inputs,
    ) -> T:

        config = config.copy()
        config['pool_size'] = pool_size
        config['strides'] = strides

        return layer_factory(cls._default_config, input, config)
