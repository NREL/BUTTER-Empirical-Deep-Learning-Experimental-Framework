from typing import Any, Callable, List, Optional, Tuple, Union
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import dmp.keras_interface.keras_keys as keras_keys

class ConvolutionalKerasLayer(layers.Layer):

    def __init__(
        self,
        conv_layer_factory: Callable,
        batch_normalizer: Callable,
        activation: Callable,
        **conv_layer_args,
    ):
        conv_layer_args[keras_keys.activation] = keras.activations.linear
        self.conv_layer = conv_layer_factory(**conv_layer_args)
        self.batch_normalizer = batch_normalizer
        self.activation = activation
        super().__init__()

    def call(self, input):
        # see https://stackoverflow.com/questions/55827660/batchnormalization-implementation-in-keras-tf-backend-before-or-after-activa
        x = self.conv_layer(input)
        x = self.batch_normalizer(x)
        return self.activation(x)