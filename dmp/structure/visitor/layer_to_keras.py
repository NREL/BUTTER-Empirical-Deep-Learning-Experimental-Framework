from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeAlias, TypeVar, Union
import tensorflow.keras as keras
from tensorflow import Tensor
from cnn.cell_structures import DenseConvolutionalLayer, SeparableConvolutionalLayer
from dmp.structure.layer import Layer
from dmp.structure.layer_visitor import LayerVisitor
from dmp.task.aspect_test.aspect_test_utils import make_from_typed_config

KerasLayer: TypeAlias = Any


class LayerToKerasVisitor(LayerVisitor[KerasLayer]):

    def __init__(self, target: Layer) -> None:

        self._inputs: list = []
        self._layer_to_keras_map: Dict[Layer, KerasLayer] = {}
        self._outputs: list = []

        output = self._convert(target)
        self._outputs = [output]

    def __call__(self) -> Tuple[list, list, Dict[Layer, KerasLayer]]:
        return self._inputs, self._outputs, self._layer_to_keras_map

    def _convert(self, target: Layer) -> KerasLayer:
        if target not in self._layer_to_keras_map:
            keras_inputs = \
                [ci for ci in (self._convert(i) for i in target.inputs)
                if ci is not None]

            keras_layer = self._visit(
                target,
                target.config.copy(),
                keras_inputs,
            )

            self._layer_to_keras_map[target] = keras_layer
        return self._layer_to_keras_map[target]

    def _visit_Input(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        result = keras.Input(**config)
        self._inputs.append(result)
        return result

    def _visit_Dense(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        _setup_keras_regularizers(config)
        return keras.layers.Dense(**config)(*keras_inputs)

    def _visit_add(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        return keras.layers.add(keras_inputs)

    def _visit_Concatenate(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        return keras.layers.Concatenate(**config)(*keras_inputs)

    def _visit_DenseConvolutionalLayer(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        _setup_keras_regularizers(config)
        return DenseConvolutionalLayer(**config)(*keras_inputs)

    def _visit_SeparableConvolutionalLayer(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        _setup_keras_regularizers(config)
        return SeparableConvolutionalLayer(**config)(*keras_inputs)

    def _visit_MaxPool2D(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        return keras.layers.MaxPool2D(**config)(*keras_inputs)

    def _visit_GlobalAveragePooling2D(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        return keras.layers.GlobalAveragePooling2D(**config)(*keras_inputs)

    def _visit_IdentityOperation(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        return keras_inputs[0]

    def _visit_ZeroizeOperation(
        self,
        target: Layer,
        config: Dict,
        keras_inputs: List,
    ) -> KerasLayer:
        return None

def make_keras_network_from_layer(source:Layer) \
    -> Tuple[list, list, Dict[Layer, KerasLayer], ]:
    return LayerToKerasVisitor(source)()


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


def _setup_keras_regularizer(
        config: Dict, key: str) -> Optional[keras.regularizers.Regularizer]:
    return _setup_in_config(
        config,
        key,
        {
            'l1': keras.regularizers.L1,
            'l2': keras.regularizers.L2,
            'l1l2': keras.regularizers.L1L2,
        },
        'regularizer',
    )


def _setup_keras_regularizers(config: Dict) -> None:
    _setup_keras_regularizer(config, 'kernel_regularizer')
    _setup_keras_regularizer(config, 'bias_regularizer')
    _setup_keras_regularizer(config, 'activity_regularizer')
