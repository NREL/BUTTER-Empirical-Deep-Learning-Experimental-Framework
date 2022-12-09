from functools import singledispatchmethod
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union
import tensorflow.keras as keras
import tensorflow
from dmp.layer.visitor.keras_interface.convolutional_layer import ConvolutionalLayer
from dmp.model.keras_layer_info import KerasLayer, KerasLayerInfo
from dmp.model.model_info import ModelInfo
from dmp.model.network_info import NetworkInfo
from dmp.model.keras_network_info import KerasNetworkInfo
from dmp.task.task_util import get_params_and_type_from_config
import dmp.task.aspect_test.keras_utils as keras_utils
from dmp.layer import *


class LayerToKerasVisitor:

    def __init__(self, target: Layer) -> None:
        self._info: KerasNetworkInfo = KerasNetworkInfo({}, [], [])
        self._info.outputs.append(
            self._make_keras_network(target).output_tensor)

    def __call__(self) -> KerasNetworkInfo:
        return self._info

    def _make_keras_network(self, target: Layer) -> KerasLayerInfo:
        keras_map = self._info.layer_to_keras_map
        if target in keras_map:
            return keras_map[target]

        result = self._visit(
            target,
            target.config.copy(),
            [(self._make_keras_network(i).output_tensor
              for i in target.inputs)],
        )

        keras_map[target] = result
        return result

    @singledispatchmethod
    def _visit(
        self,
        target: Layer,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        raise NotImplementedError(f'Unknown Layer of type {type(target)}.')

    @_visit.register
    def _(
        self,
        target: Input,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        result = keras.Input(**config)
        self._info.inputs.append(result)  # type: ignore
        return KerasLayerInfo(target, result, result)  # type: ignore

    @_visit.register
    def _(
        self,
        target: Dense,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        _make_keras_regularizers(config)
        _make_keras_activation(config)
        return _make_keras_layer(target, keras.layers.Dense, config, inputs)

    @_visit.register
    def _(
        self,
        target: Add,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_keras_layer(target, keras.layers.Add, config, inputs)

    @_visit.register
    def _(
        self,
        target: Concatenate,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_keras_layer(target, keras.layers.Concatenate, config,
                                 inputs)

    @_visit.register
    def visit_DenseConvolutionalLayer(
        self,
        target: DenseConv,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_convolutional_layer(
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
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_convolutional_layer(
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
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        config['padding'] = 'same'
        config['activation'] = keras.activations.linear
        config['kernel_size'] = (1, ) * target.dimension
        return self.visit_DenseConvolutionalLayer(
            target,
            config,
            inputs,
        )

    @_visit.register
    def _(
        self,
        target: MaxPool,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_by_dimension(
            target, config, inputs, {
                1: keras.layers.MaxPool1D,
                2: keras.layers.MaxPool2D,
                3: keras.layers.MaxPool3D,
            })

    @_visit.register
    def _(
        self,
        target: AvgPool,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_by_dimension(
            target, config, inputs, {
                1: keras.layers.AvgPool1D,
                2: keras.layers.AvgPool2D,
                3: keras.layers.AvgPool3D,
            })

    @_visit.register
    def _(
        self,
        target: GlobalAveragePooling,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_by_dimension(
            target, config, inputs, {
                1: keras.layers.GlobalAveragePooling1D,
                2: keras.layers.GlobalAveragePooling2D,
                3: keras.layers.GlobalAveragePooling3D,
            })

    @_visit.register
    def _(
        self,
        target: GlobalMaxPooling,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return _make_by_dimension(
            target, config, inputs, {
                1: keras.layers.GlobalMaxPool1D,
                2: keras.layers.GlobalMaxPool2D,
                3: keras.layers.GlobalMaxPool3D,
            })

    @_visit.register
    def _(
        self,
        target: IdentityOperation,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        return self._info.layer_to_keras_map[target.input]

    @_visit.register
    def _(
        self,
        target: ZeroizeOperation,
        config: Dict[str, Any],
        inputs: List[KerasLayer],
    ) -> KerasLayerInfo:
        keras_layer = tensorflow.zeros_like(target.shape)
        return KerasLayerInfo(target, keras_layer, keras_layer)


def make_keras_network_from_layer(target: Layer) -> KerasNetworkInfo:
    return LayerToKerasVisitor(target)()


def make_keras_model_from_network(network: NetworkInfo) -> ModelInfo:
    keras_network = make_keras_network_from_layer(network.structure)
    keras_model = keras.Model(
        inputs=keras_network.inputs,
        outputs=keras_network.outputs,
    )
    if len(keras_model.inputs) != 1:  # type: ignore
        raise ValueError('Wrong number of keras inputs generated')

    if keras_utils.count_trainable_parameters_in_keras_model(keras_model)\
        != network.num_free_parameters:
        raise RuntimeError('Wrong number of trainable parameters')

    return ModelInfo(network, keras_network, keras_model)


def _make_by_dimension(
    target: Layer,
    config: Dict[str, Any],
    inputs: List[KerasLayer],
    dimension_to_factory_map: Dict[int, Callable],
) -> KerasLayerInfo:
    keras_class = dimension_to_factory_map[target.dimension]
    return _make_keras_layer(target, keras_class, config, inputs)


def _make_convolutional_layer(
    target: ConvolutionalLayer,
    config: Dict[str, Any],
    inputs: List[KerasLayer],
    dimension_to_factory_map: Dict[int, Callable],
) -> KerasLayerInfo:
    _make_keras_regularizers(config)
    _make_keras_activation(config)
    _make_keras_batch_normalization(config)
    config['conv_layer_factory'] = \
        dimension_to_factory_map[target.dimension]
    return _make_keras_layer(target, ConvolutionalLayer, config, inputs)


def _make_activation_from_config(config: Dict[str, Any]) -> Callable:
    type, params = get_params_and_type_from_config(config)
    activation_function = keras.activations.get(type)
    if activation_function is None:
        raise ValueError(f'Unknown activation {config}.')
    return lambda x: activation_function(x, **params)


def _make_keras_regularizer(
        config: Dict[str, Any]) -> keras.regularizers.Regularizer:
    return _make_from_config_using_keras_get(
        config,
        keras.regularizers.get,
        'regularizer',
    )


def make_in_config(config: Dict[str, Any], key: str,
                   factory: Callable) -> None:
    config[key] = factory(config.get(key, None))


def _make_keras_regularizers(config: Dict[str, Any]) -> None:
    make_in_config(config, 'kernel_regularizer', _make_keras_regularizer)
    make_in_config(config, 'bias_regularizer', _make_keras_regularizer)
    make_in_config(config, 'activity_regularizer', _make_keras_regularizer)


def _make_keras_activation(config: Dict[str, Any]) -> None:
    make_in_config(config, 'activation', _make_activation_from_config)


def _make_batch_norm_from_config(config: Optional[Dict[str, Any]]) -> Any:
    if config is None:
        return lambda x: x
    return keras.layers.BatchNormalization(**config)


def _make_keras_batch_normalization(config: Dict[str, Any]) -> None:
    make_in_config(config, 'batch_normalization', _make_batch_norm_from_config)


def _make_keras_layer(
    layer: Layer,
    target: Callable,
    config: Dict[str, Any],
    inputs: List[KerasLayer],
) -> KerasLayerInfo:
    keras_layer = target(**config)
    keras_output = keras_layer(*inputs)
    return KerasLayerInfo(layer, keras_layer, keras_output)  # type: ignore



def _make_from_config_using_keras_get(
        config: dict,
        keras_get_function: Callable,
        name: str,  # used for exception messages
) -> Any:
    if config is None:
        return None

    type, params = get_params_and_type_from_config(config)
    result = keras_get_function({'class_name': type, 'config': params})
    if result is None:
        raise ValueError(f'Unknown {name}, {config}.')

