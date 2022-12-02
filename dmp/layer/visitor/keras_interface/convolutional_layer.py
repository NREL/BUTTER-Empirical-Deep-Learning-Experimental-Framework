from typing import Any, Callable, List, Tuple, Union
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from dmp.task.aspect_test.aspect_test_utils import get_activation_factory, get_batch_normalization_factory


class ConvolutionalLayer(layers.Layer):

    def __init__(
        self,
        conv_layer_factory: Callable,
        batch_norm: str,
        activation: str,
        **conv_layer_args,
    ):
        conv_layer_args['activation'] = keras.activations.linear
        self.conv_layer = conv_layer_factory(**conv_layer_args)
        self.batch_norm = batch_norm
        self.batch_normalizer = \
            get_batch_normalization_factory(self.batch_norm)
        self.activation_function = get_activation_factory(activation)
        super().__init__()

    def call(self, input):
        # see https://stackoverflow.com/questions/55827660/batchnormalization-implementation-in-keras-tf-backend-before-or-after-activa
        x = self.conv_layer(input)
        x = self.batch_normalizer(x)
        return self.activation_function(x)