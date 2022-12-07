from functools import singledispatchmethod
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union
import tensorflow.keras as keras
import tensorflow
from dmp.layer.visitor.keras_interface.convolutional_layer import ConvolutionalLayer
from dmp.task.task_util import get_params_and_type_from_config, make_from_config_using_keras_get, make_from_optional_typed_config, make_from_typed_config, make_typed_config_factory
from dmp.layer import *

KerasLayer = Union[keras.layers.Layer, tensorflow.Tensor]


class LayerToKerasVisitor:

    def __init__(
        self,
        target: Layer,
        layer_shapes: Dict[Layer, Tuple],
    ) -> None:
        self._layer_shapes: Dict[Layer, Tuple] = layer_shapes

        self._inputs: list = []
        self._layer_to_keras_map: Dict[Layer, Tuple[KerasLayer,
                                                    tensorflow.Tensor]] = {}
        self._outputs: list = []

        output = self._get_keras_layer(target)
        self._outputs = [output]

    def __call__(
        self
    ) -> Tuple[list, list, Dict[Layer, Tuple[KerasLayer, tensorflow.Tensor]]]:
        return self._inputs, self._outputs, self._layer_to_keras_map

    def _get_keras_layer(
            self, target: Layer) -> Tuple[KerasLayer, tensorflow.Tensor]:
        if target in self._layer_to_keras_map:
            return self._layer_to_keras_map[target]

        inputs = \
            [keras_layer
            for keras_layer, keras_output in
            (self._get_keras_layer(i) for i in target.inputs)]

        result = self._visit(
            target,
            target.config.copy(),
            inputs,
        )

        self._layer_to_keras_map[target] = result
        return result

    def _output_shape(self, target: Layer) -> Tuple:
        return self._layer_shapes[target]

    def _dimension(self, target: Layer) -> int:
        return len(self._output_shape(target)) - 1

    def _make_by_dimension(
        self,
        target: Layer,
        config: Dict,
        inputs: List[KerasLayer],
        dimension_to_factory_map: Dict[int, Callable],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        keras_class = dimension_to_factory_map[self._dimension(target)]
        return _make_keras(keras_class, config, inputs)

    @singledispatchmethod
    def _visit(
        self,
        target: Layer,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        raise NotImplementedError(f'Unknown Layer of type {type(target)}.')

    @_visit.register
    def _(
        self,
        target: Input,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        result = keras.Input(**config)
        self._inputs.append(result)
        return result, result  # type: ignore

    @_visit.register
    def _(
        self,
        target: Dense,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        _make_keras_regularizers(config)
        _make_keras_activation(config)
        return _make_keras(keras.layers.Dense, config, inputs)

    @_visit.register
    def _(
        self,
        target: Add,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return _make_keras(keras.layers.Add, config, inputs)

    @_visit.register
    def _(
        self,
        target: Concatenate,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return _make_keras(keras.layers.Concatenate, config, inputs)

    @_visit.register
    def visit_DenseConvolutionalLayer(
        self,
        target: DenseConv,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return self._make_convolutional_layer(
            target,
            config,
            inputs,
            {
                1: keras.layers.Conv1D,
                2: keras.layers.Conv2D,
                3: keras.layers.Conv3D,
            },
        )

    @_visit.register
    def _(
        self,
        target: SeparableConv,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return self._make_convolutional_layer(
            target,
            config,
            inputs,
            {
                1: keras.layers.SeparableConv1D,
                2: keras.layers.SeparableConv2D,
            },
        )

    @_visit.register
    def _(
        self,
        target: ProjectionOperation,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        config['padding'] = 'same'
        config['activation'] = keras.activations.linear
        config['kernel_size'] = (1, ) * self._dimension(target)
        return self.visit_DenseConvolutionalLayer(
            target,
            config,
            inputs,
        )

    @_visit.register
    def _(
        self,
        target: MaxPool,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return self._make_by_dimension(
            target, config, inputs, {
                1: keras.layers.MaxPool1D,
                2: keras.layers.MaxPool2D,
                3: keras.layers.MaxPool3D,
            })

    @_visit.register
    def _(
        self,
        target: AvgPool,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return self._make_by_dimension(
            target, config, inputs, {
                1: keras.layers.AvgPool1D,
                2: keras.layers.AvgPool2D,
                3: keras.layers.AvgPool3D,
            })

    @_visit.register
    def _(
        self,
        target: GlobalAveragePooling,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return self._make_by_dimension(
            target, config, inputs, {
                1: keras.layers.GlobalAveragePooling1D,
                2: keras.layers.GlobalAveragePooling2D,
                3: keras.layers.GlobalAveragePooling3D,
            })

    @_visit.register
    def _(
        self,
        target: GlobalMaxPooling,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        return self._make_by_dimension(
            target, config, inputs, {
                1: keras.layers.GlobalMaxPool1D,
                2: keras.layers.GlobalMaxPool2D,
                3: keras.layers.GlobalMaxPool3D,
            })

    @_visit.register
    def _(
        self,
        target: IdentityOperation,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        keras_layer = inputs[0]
        return keras_layer, self._layer_to_keras_map[target][1]

    @_visit.register
    def _(
        self,
        target: ZeroizeOperation,
        config: Dict,
        inputs: List[KerasLayer],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        keras_layer = tensorflow.zeros_like(self._output_shape(target))
        return keras_layer, keras_layer

    def _make_convolutional_layer(
        self,
        target: AConvolutionalLayer,
        config: Dict,
        inputs: List[KerasLayer],
        dimension_to_factory_map: Dict[int, Callable],
    ) -> Tuple[KerasLayer, tensorflow.Tensor]:
        _make_keras_regularizers(config)
        _make_keras_activation(config)
        _make_keras_batch_normalization(config)
        config['conv_layer_factory'] = \
            dimension_to_factory_map[self._dimension(target)]
        return _make_keras(ConvolutionalLayer, config, inputs)


def make_keras_network_from_layer(
    target: Layer,
    layer_shapes: Dict[Layer, Tuple],
) -> Tuple[list, list, Dict[Layer, Tuple[KerasLayer, tensorflow.Tensor]], ]:
    return LayerToKerasVisitor(target, layer_shapes)()


def _setup_in_config(
    config: Dict,
    key: str,
    mapping: Dict[str, Callable],
    config_name: str,
    *args,
    **kwargs,
):
    if key not in config:
        return None

    result = make_from_typed_config(
        config[key],
        mapping,
        config_name,
        *args,
        **kwargs,
    )
    config[key] = result
    return result


def _make_activation_from_config(config: dict) -> Callable:
    type, params = get_params_and_type_from_config(config)
    activation_function = keras.activations.get(type)
    if activation_function is None:
        raise ValueError(f'Unknown activation {config}.')
    return lambda x: activation_function(x, **params)


# def _make_activation_factory(activation_function: Callable) -> Callable:
#     return lambda **params: (lambda x: activation_function(x, **params))
# _make_activation_from_config = make_typed_config_factory(
#     'activation', {
#         k: _make_activation_factory(v)
#         for k, v in {
#             'elu': keras.activations.elu,
#             'exponential': keras.activations.exponential,
#             'gelu': keras.activations.gelu,
#             'hard_sigmoid': keras.activations.hard_sigmoid,
#             'linear': keras.activations.linear,
#             'relu': keras.activations.relu,
#             'selu': keras.activations.selu,
#             'sigmoid': keras.activations.sigmoid,
#             'softmax': keras.activations.softmax,
#             'softplus': keras.activations.softplus,
#             'softsign': keras.activations.softsign,
#             'swish': keras.activations.swish,
#             'tanh': keras.activations.tanh,
#         }.items()
#     })


def _make_keras_regularizer(config: dict) -> keras.regularizers.Regularizer:
    return make_from_config_using_keras_get(
        config,
        keras.regularizers.get,
        'regularizer',
    )


# _make_keras_regularizer = make_typed_config_factory(
#     'regularizer', {
#         'L1': keras.regularizers.L1,
#         'L2': keras.regularizers.L2,
#         'L1L2': keras.regularizers.L1L2,
#     })


def make_in_config(config: dict, key: str, factory: Callable) -> None:
    config[key] = factory(config.get(key, None))


def _make_keras_regularizers(config: dict) -> None:
    make_in_config(config, 'kernel_regularizer', _make_keras_regularizer)
    make_in_config(config, 'bias_regularizer', _make_keras_regularizer)
    make_in_config(config, 'activity_regularizer', _make_keras_regularizer)


def _make_keras_activation(config: dict) -> None:
    make_in_config(config, 'activation', _make_activation_from_config)


def _make_batch_norm_from_config(config: Optional[dict]) -> Any:
    if config is None:
        return lambda x: x
    return keras.layers.BatchNormalization(**config)


def _make_keras_batch_normalization(config: dict) -> None:
    make_in_config(config, 'batch_normalization', _make_batch_norm_from_config)


def _make_keras(
    target: Callable,
    config: Dict,
    inputs: List[KerasLayer],
) -> Tuple[KerasLayer, tensorflow.Tensor]:
    keras_layer = target(**config)
    keras_output = keras_layer(*inputs)
    return keras_layer, keras_output
