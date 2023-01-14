
from typing import List, Union
from dmp.layer.layer import Layer, LayerConfig, empty_config, empty_inputs


class Dense(Layer):

    _default_config: LayerConfig = {
        'activation': 'relu',
        'use_bias': True,
        'kernel_initializer': 'HeUniform',
        'bias_initializer': 'Zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
    }

    @staticmethod
    def make(
        units: int,
        config: LayerConfig = empty_config,
        input: List[Layer] = empty_inputs,
    ) -> 'Dense':
        config['units'] = units
        return Dense(Dense._default_config, input, config)

