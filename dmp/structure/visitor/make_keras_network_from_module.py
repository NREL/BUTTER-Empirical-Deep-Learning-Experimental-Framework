from functools import singledispatchmethod
from typing import Any, Dict, List, Tuple
from cnn.cell_structures import DenseConvolutionalLayer, IdentityOperation, SeparableConvolutionalLayer, ZeroizeOperation
from dmp.structure.n_conv import NConcat, NConv, NGlobalPool, NIdentity, NMaxPool, NSepConv, NZeroize
from dmp.structure.network_module import NetworkModule
from dmp.structure.n_add import NAdd
from dmp.structure.n_dense import NDense
from dmp.structure.n_input import NInput

import tensorflow.keras as keras

from dmp.task.aspect_test.aspect_test_utils import setup_keras_regularizer


class MakeKerasNetworkFromModuleVisitor:
    """
    Visitor that makes a keras module for a given network module and input keras modules
    """

    def __init__(self, target: NetworkModule) -> None:
        self._inputs: list = []
        self._node_layer_map: Dict[NetworkModule, keras.layers.Layer] = {}
        self._outputs: list = []

        output = self._visit(target)
        self._outputs = [output]

    def __call__(
            self
    ) -> Tuple[list, list, Dict[NetworkModule, keras.layers.Layer], ]:
        return self._inputs, self._outputs, self._node_layer_map

    def _visit(self, target: NetworkModule) -> keras.layers.Layer:
        if target not in self._node_layer_map:
            keras_inputs = [self._visit(i) for i in target.inputs]
            keras_layer = self._visit_raw(target, keras_inputs)
            self._node_layer_map[target] = keras_layer
        return self._node_layer_map[target]

    @singledispatchmethod
    def _visit_raw(self, target, keras_inputs) -> Any:
        raise NotImplementedError(
            f'Unsupported module of type "{type(target)}".')

    @_visit_raw.register
    def _(self, target: NInput, keras_inputs) -> Any:
        result = keras.Input(shape=target.shape)
        self._inputs.append(result)
        return result

    @_visit_raw.register
    def _(self, target: NDense, keras_inputs) -> Any:

        kernel_regularizer = setup_keras_regularizer(target.kernel_regularizer)
        bias_regularizer = setup_keras_regularizer(target.bias_regularizer)
        activity_regularizer = setup_keras_regularizer(
            target.activity_regularizer)

        return keras.layers.Dense(
            target.output_size,
            activation=target.activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_initializer=target.kernel_initializer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NAdd, keras_inputs) -> Any:
        return keras.layers.add(keras_inputs)

    @_visit_raw.register
    def _(self, target: NConcat, keras_inputs) -> Any:
        return keras.layers.Concatenate()(keras_inputs)

    @_visit_raw.register
    def _(self, target: NConv, keras_inputs) -> Any:
        kernel_regularizer = setup_keras_regularizer(target.kernel_regularizer)
        bias_regularizer = setup_keras_regularizer(target.bias_regularizer)
        activity_regularizer = setup_keras_regularizer(
            target.activity_regularizer)

        return DenseConvolutionalLayer(
            kernel_size=target.kernel_size,
            filters=target.filters,
            activation=target.activation,
            batch_norm=target.batch_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NSepConv, keras_inputs) -> Any:
        kernel_regularizer = setup_keras_regularizer(target.kernel_regularizer)
        bias_regularizer = setup_keras_regularizer(target.bias_regularizer)
        activity_regularizer = setup_keras_regularizer(
            target.activity_regularizer)

        return SeparableConvolutionalLayer(
            kernel_size=target.kernel_size,
            filters=target.filters,
            activation=target.activation,
            batch_norm=target.batch_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NMaxPool, keras_inputs) -> Any:
        return keras.layers.MaxPool2D(pool_size=target.kernel_size,
                                      strides=target.stride,
                                      padding=target.padding)(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NGlobalPool, keras_inputs) -> Any:
        return keras.layers.GlobalAveragePooling2D()(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NIdentity, keras_inputs) -> Any:
        return IdentityOperation()(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NZeroize, keras_inputs) -> Any:
        return ZeroizeOperation()(*keras_inputs)


def make_keras_network_from_network_module(
    source: NetworkModule,
) -> Tuple[list, list, Dict[NetworkModule, keras.layers.Layer], ]:
    return MakeKerasNetworkFromModuleVisitor(source)()