from abc import ABC
from typing import Callable, Dict, Optional, Tuple, Any, List, Sequence, TypeVar, Union
from dmp.layer.spatitial_layer import SpatitialLayer
from dmp.layer.layer import Layer, LayerConstructor, LayerConfig, empty_config, empty_inputs
import dmp.keras_interface.keras_keys as keras_keys

T = TypeVar('T')


class PoolingLayer(SpatitialLayer, ABC):

    @property
    def strides(self) -> Tuple:
        config = self.config
        strides = config.get(keras_keys.strides, None)
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
        config[keras_keys.pool_size] = pool_size
        config[keras_keys.strides] = strides

        return layer_factory(cls._default_config, input, config)
