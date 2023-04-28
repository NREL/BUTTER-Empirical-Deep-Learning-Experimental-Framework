from dataclasses import dataclass, field
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
from dmp.structure.batch_norm_block import BatchNormBlock

@dataclass
class ResNetBlock(LayerFactory):
    num_filters: int
    stride: int
    batch_normalization: BatchNormalization = field(default_factory=BatchNormalization)
    config: LayerConfig = field(
        default_factory=lambda: {
            'padding': 'same',
            'use_bias': False,
            'activation': 'relu',
        }
    )

    def make_layer(
        self,
        overrides: LayerConfig,
        inputs: Union['Layer', List['Layer']],
    ) -> Layer:
        '''
        residual -> conv1 -> conv2 ->  add -> relu ->
                -> [downsample]   ->
        '''
        stride = self.stride

        conv1 = BatchNormBlock(
            DenseConv.make(
                self.num_filters,
                [3, 3],
                [stride, stride],
                self.config,
            ),
            self.batch_normalization,
        ).make_layer(overrides, inputs)

        linear_overrides = overrides.copy()
        linear_overrides['activation'] = 'linear'

        conv2 = BatchNormBlock(
            DenseConv.make(
                self.num_filters,
                [3, 3],
                [1, 1],
                self.config,
            ),
            self.batch_normalization,
        ).make_layer(linear_overrides, conv1)

        residual = inputs[0] if not isinstance(inputs, Layer) else inputs
        if stride > 1:
            residual = BatchNormBlock(
                DenseConv.make(
                    self.num_filters,
                    [1, 1],
                    [stride, stride],
                    self.config,
                ),
                self.batch_normalization,
            ).make_layer(linear_overrides, residual)

        result = Add({}, [conv2, residual])

        activation = overrides.get(
            'activation', self.config.get('activation', 'linear')
        )
        if activation != 'linear':
            result = OpLayer.make(activation, {}, result)
        return result
