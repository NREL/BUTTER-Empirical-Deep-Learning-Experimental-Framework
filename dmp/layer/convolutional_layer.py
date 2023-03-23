from abc import ABC
from typing import Any, Dict, Optional, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.keras_interface.keras_utils import make_keras_config
from dmp.layer.layer import Layer, LayerConfig, LayerConstructor, empty_config, empty_inputs
from dmp.layer.spatitial_layer import SpatitialLayer

T = TypeVar('T')


class ConvolutionalLayer(SpatitialLayer, ABC):

    _default_config = {
        'strides': (1, 1),
        'padding': 'valid',
        'data_format': None,
        'dilation_rate': (1, 1),
        'groups': 1,
        'activation': 'relu',
        'use_bias': True,
        'kernel_initializer': 'HeUniform',
        'bias_initializer': 'Zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
        'batch_normalizer': None,
    }

    @classmethod
    def make(
        cls,
        layer_factory: LayerConstructor[T],
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: LayerConfig = empty_config,
        inputs: List[Layer] = empty_inputs,
    ) -> T:
        config = config.copy()
        config.update({
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
        })
        return layer_factory(
            cls._default_config,
            inputs,
            config,
        )
