from dataclasses import dataclass, field
from typing import List, Optional, Union
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.layer.batch_normalization import BatchNormalization
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import Layer, LayerConfig, LayerFactory
from dmp.layer.add import Add
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense_conv import DenseConv
from dmp.layer.flatten import Flatten
from dmp.layer.global_average_pooling import GlobalAveragePooling
from dmp.layer.identity import Identity
from dmp.layer.max_pool import MaxPool
from dmp.layer.op_layer import OpLayer


@dataclass
class BatchNormBlock(LayerFactory):
    normalized_layer: Layer
    batch_normalization: Optional[BatchNormalization] = field(
        default_factory=BatchNormalization
    )

    def make_layer(
        self,
        config: LayerConfig,
        inputs: Union['Layer', List['Layer']],
    ) -> Layer:
        output = self.normalized_layer.make_layer(config, inputs)
        activation_function = output.config.pop('activation')
        output['activation'] = 'linear'

        if self.batch_normalization is not None:
            output = self.batch_normalization.make_layer(config, output)

        if activation_function is not None and activation_function != 'linear':
            output = OpLayer.make(activation_function, {}, output)

        return output
