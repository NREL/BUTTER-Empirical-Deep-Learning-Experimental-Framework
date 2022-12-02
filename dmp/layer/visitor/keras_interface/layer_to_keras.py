from functools import singledispatchmethod
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union
import tensorflow.keras as keras
import tensorflow
from dmp.layer.visitor.keras_interface.convolutional_layer import ConvolutionalLayer
from dmp.task.aspect_test.aspect_test_utils import make_from_typed_config
from dmp.layer import *
# from dmp.cnn.cell_structures import ConvolutionalLayer

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
        raise NotImplementedError(f'Unsupported Layer of type {type(target)}.')

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
        _setup_keras_regularizers(config)
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
        config['activation'] = 'linear'
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
        _setup_keras_regularizers(config)
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


def _make_keras(
    target: Callable,
    config: Dict,
    inputs: List[KerasLayer],
) -> Tuple[KerasLayer, tensorflow.Tensor]:
    keras_layer = target(**config)
    keras_output = keras_layer(*inputs)
    return keras_layer, keras_output
