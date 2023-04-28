from abc import ABC
from typing import Any, Dict, Optional, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.keras_interface.keras_utils import make_keras_config
from dmp.layer.layer import Layer, LayerConfig, LayerConstructor, empty_config, empty_inputs
from dmp.layer.spatitial_layer import SpatitialLayer
import dmp.keras_interface.keras_keys as keras_keys

T = TypeVar('T')


class ConvolutionalLayer(SpatitialLayer, ABC):

    _default_config = {
        keras_keys.strides: [1, 1],
        keras_keys.padding: 'valid',
        keras_keys.data_format: None,
        keras_keys.dilation_rate: [1, 1],
        keras_keys.groups: 1,
        keras_keys.activation: 'relu',
        keras_keys.use_bias: True,
        keras_keys.kernel_initializer: 'HeUniform',
        keras_keys.bias_initializer: 'Zeros',
        keras_keys.kernel_regularizer: None,
        keras_keys.bias_regularizer: None,
        keras_keys.activity_regularizer: None,
        keras_keys.kernel_constraint: None,
        keras_keys.bias_constraint: None,
        # keras_keys.batch_normalization: None,
    }

    @classmethod
    def make(
        cls,
        layer_factory: LayerConstructor[T],
        filters: int,
        kernel_size: List[int],
        strides: List[int],
        config: LayerConfig = empty_config,
        inputs: Union['Layer', List['Layer']] = empty_inputs,
    ) -> T:
        config = config.copy()
        config.update({
            keras_keys.filters: filters,
            keras_keys.kernel_size: kernel_size,
            keras_keys.strides: strides,
        })
        
        return layer_factory(
            cls._default_config,
            inputs,
            config,
        )
