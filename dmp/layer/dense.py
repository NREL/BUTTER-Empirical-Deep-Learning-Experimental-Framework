
from typing import List, Union
from dmp.layer.layer import Layer, LayerConfig, empty_config, empty_inputs
import dmp.keras_interface.keras_keys as keras_keys

class Dense(Layer):

    _default_config: LayerConfig = {
        keras_keys.activation: 'relu',
        keras_keys.use_bias: True,
        keras_keys.kernel_initializer: 'HeUniform',
        keras_keys.bias_initializer: 'Zeros',
        keras_keys.kernel_regularizer: None,
        keras_keys.bias_regularizer: None,
        keras_keys.activity_regularizer: None,
        keras_keys.kernel_constraint: None,
        keras_keys.bias_constraint: None,
    }

    @staticmethod
    def make(
        units: int,
        config: LayerConfig = empty_config,
        input: List[Layer] = empty_inputs,
    ) -> 'Dense':
        config['units'] = units
        return Dense(Dense._default_config, input, config)

