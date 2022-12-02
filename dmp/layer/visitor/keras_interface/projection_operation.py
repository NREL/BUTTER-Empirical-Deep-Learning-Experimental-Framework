from typing import Any, Callable, List, Tuple, Union
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from dmp.task.aspect_test.aspect_test_utils import get_activation_factory, get_batch_normalization_factory


class ProjectionOperation(layers.Layer):

    def __init__(
        self,
        filters=128,
        batch_norm='none',
        activation='relu',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super(ProjectionOperation, self).__init__()
        # batch norm and activation are not used in projection operation
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(
            filters,
            1,
            padding='same',
            activation='linear',
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )

    def call(self, x):
        x = self.conv(x)
        return x