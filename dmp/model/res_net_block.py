from dataclasses import dataclass
from typing import List, Union
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.layer.batch_normalization import BatchNormalization
from dmp.layer.layer import Layer, LayerConfig, LayerFactory
from dmp.layer.add import Add
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense_conv import DenseConv
from dmp.layer.flatten import Flatten
from dmp.layer.global_average_pooling import GlobalAveragePooling
from dmp.layer.identity import Identity
from dmp.layer.max_pool import MaxPool
from dmp.layer.op_layer import OpLayer

'''
{
                'padding': 'same',
                'use_bias': False,
                'batch_normalization': make_keras_kwcfg('BatchNormalization'),
                'activation': 'relu',
            }
'''


@dataclass
class ResNetBlock(LayerFactory):
    num_filters: int
    stride: int

    _default_config = {
        'padding': 'same',
        'use_bias': False,
        'batch_normalization': BatchNormalization(),
        'activation': 'relu',
    }

    def make_layer(
        self,
        inputs: Union['Layer', List['Layer']],
        config: LayerConfig,
    ) -> Layer:
        '''
        residual -> conv1 -> conv2 ->  add -> relu ->
                -> [downsample]   ->
        '''
        c = self._default_config.copy()
        c.update(config)
        config = c

        stride = self.stride

        conv1 = DenseConv.make(
            self.num_filters,
            [3, 3],
            [stride, stride],
            config,
            inputs,
        )

        inner_config = config.copy()
        inner_config['activation'] = 'linear'
        inner_config['batch_normalization'] = None

        conv2 = DenseConv.make(
            self.num_filters,
            [3, 3],
            [1, 1],
            inner_config,
            conv1,
        )

        residual = inputs[0] if not isinstance(inputs, Layer) else inputs
        if stride > 1:
            residual = DenseConv.make(
                self.num_filters,
                [1, 1],
                [stride, stride],
                inner_config,
                residual,
            )

        result = Add({}, [residual, conv2])
        batch_normalization = config.get('batch_normalization', None)
        if batch_normalization is not None:
            result = batch_normalization.make_layer(result, {})
        result = OpLayer.make('relu', {}, result)
        return result
