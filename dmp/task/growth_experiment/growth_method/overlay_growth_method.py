from functools import singledispatchmethod
from typing import Any, Dict, Iterable, Set, Tuple, TypeVar

import numpy
import tensorflow.keras as keras
from dmp.layer import *
from dmp.layer.visitor.keras_interface.layer_to_keras import KerasLayer
from dmp.model.model_info import ModelInfo
from dmp.task.growth_experiment.growth_method.growth_method import GrowthMethod
from dmp.task.growth_experiment.layer_growth_info import LayerGrowthInfo

T = TypeVar('T')


class OverlayGrowthMethod(GrowthMethod):
    """
    Visitor that fills one network with the values from another. 
    If the destination network is larger, this will 'grow' the src into
    the destination.
    """

    def __init__(
        self,
        old_to_old_scale: float = 1.0,
        new_to_new_scale: float = 1.0,
        old_to_new_scale: float = 1.0,
        new_to_old_scale: float = 0.0,
        new_add_to_old_scale: float = 0.0,
    ) -> None:
        self.old_to_old_scale: float = old_to_old_scale
        self.new_to_new_scale: float = new_to_new_scale
        self.old_to_new_scale: float = old_to_new_scale
        self.new_to_old_scale: float = new_to_old_scale
        self.new_add_to_old_scale: float = new_add_to_old_scale

    def grow(
        self,
        src: ModelInfo,  # 'parent' / 'previous' network
        dest: ModelInfo,  # 'child' / 'next' network
        growth_map: Dict[Layer, LayerGrowthInfo]
    ) -> None:
        for growth_info in growth_map.values():
            self._do_visit(growth_info.src.layer, growth_info)

    @singledispatchmethod
    def _do_visit(
        self,
        src_layer: Layer,
        growth_info: LayerGrowthInfo,
    ) -> None:
        raise NotImplementedError('Unsupported module of type "{}".'.format(
            type(src_layer)))

    @_do_visit.register
    def _(
        self,
        src_layer: Input,
        growth_info: LayerGrowthInfo,
    ) -> None:
        return

    @_do_visit.register
    def _(
        self,
        src_layer: Dense,
        growth_info: LayerGrowthInfo,
    ) -> None:
        self.overlay_standard_layer(src_layer, growth_info)

    @_do_visit.register
    def _(
        self,
        src_layer: DenseConv,
        growth_info: LayerGrowthInfo,
    ) -> None:
        self.overlay_standard_layer(src_layer, growth_info)

    @_do_visit.register
    def _(
        self,
        src_layer: SeparableConv,
        growth_info: LayerGrowthInfo,
    ) -> None:
        self.overlay_standard_layer(src_layer, growth_info)

    def overlay_standard_layer(
        self,
        src_layer: Layer,
        growth_info: LayerGrowthInfo,
    ) -> None:
        (
            src_keras_layer,
            src_params,
            dest_layer,
            dest_keras_layer,
            dest_params,
        ) = _get_layers(src_layer, growth_info)

        num_params = len(src_params)
        if num_params != len(dest_params):
            raise NotImplementedError(
                f'Layer parameter group numbers do not match {num_params}, {len(dest_params)}'
            )
        if num_params <= 0:
            return

        num_weight_dims = len(src_params[0].shape)
        for src_weights, dest_weights in zip(src_params, dest_params):
            num_dims = len(src_weights.shape)
            if num_dims != len(dest_weights.shape):
                raise ValueError(
                    f'Mismatched wieght shapes {src_weights.shape}, {dest_weights.shape}'
                )

            if num_dims == num_weight_dims:
                src_channels_out = src_weights.shape[-1]
                src_channels_in = src_weights.shape[-2]
                self._scale_weights(
                    src_weights,
                    dest_weights[..., :src_channels_in, :src_channels_out],
                    dest_weights[..., :src_channels_in, src_channels_out:],
                    dest_weights[..., src_channels_in:, :src_channels_out],
                    dest_weights[..., src_channels_in:, src_channels_out:],
                )
            elif num_dims == num_weight_dims - 1:
                src_biases = src_params[1]
                dest_biases = dest_params[1]
                num_src_biases = src_biases.shape[-1]
                self._scale_biases(
                    src_biases,
                    dest_biases[..., :num_src_biases],
                    dest_biases[..., num_src_biases:],
                )
            else:
                raise NotImplementedError(
                    f'Weight group dimension not supported {num_weight_dims} {num_dims}'
                )

        dest_keras_layer.set_weights(dest_params)  # type: ignore

    @_do_visit.register
    def _(
        self,
        src_layer: Add,
        growth_info: LayerGrowthInfo,
    ) -> None:
        return

    def _scale_weights(
        self,
        src_weights: numpy.ndarray,
        old_to_old: numpy.ndarray,
        old_to_new: numpy.ndarray,
        new_to_old: numpy.ndarray,
        new_to_new: numpy.ndarray,
    ):
        _blend_blocks(  # copy and scale old weights
            src_weights,
            self.old_to_old_scale,
            old_to_old,
            self.new_add_to_old_scale,
        )
        _scale_block(  # scale old to new weights
            old_to_new,
            self.old_to_new_scale,
        )
        _scale_block(  # scale new to old weights
            new_to_old,
            self.new_to_old_scale,
        )
        _scale_block(  # scale new weights
            new_to_new,
            self.new_to_new_scale,
        )

    def _scale_biases(
        self,
        src_biases: numpy.ndarray,
        old_biases: numpy.ndarray,
        new_biases: numpy.ndarray,
    ):
        _blend_blocks(  # copy and scale old biases
            src_biases,
            self.old_to_old_scale,
            old_biases,
            self.new_add_to_old_scale,
        )
        _scale_block(  # scale new biases
            new_biases,
            self.new_to_new_scale,
        )


def _scale_block(
    target: numpy.ndarray,
    scale: float,
) -> None:
    if scale == 1.0:
        pass
    elif scale == 0.0:
        target[:] = 0
    else:
        target[:] *= scale


def _blend_blocks(
    src_weights: numpy.ndarray,
    src_scale: float,
    dest_block: numpy.ndarray,
    dest_scale: float,
) -> None:
    if src_scale == 0.0:
        dest_block[:] = src_weights
        _scale_block(
            dest_block,
            src_scale,
        )
    else:
        _scale_block(
            dest_block,
            dest_scale,
        )
        dest_block[:] += src_weights * src_scale


def _as_same_type(
    src_layer: T,
    dest_layer: Any,
) -> T:
    if type(src_layer) != type(dest_layer):
        raise TypeError(
            f'src and destination Layer are not the same type {src_layer} != {dest_layer}.'
        )
    return dest_layer  # type: ignore


def _get_layers(
    src_layer: T,
    growth_info: LayerGrowthInfo,
) -> Tuple[KerasLayer, Any, T, KerasLayer, Any, ]:
    src_keras_layer = growth_info.src.keras_layer
    if not isinstance(src_keras_layer, keras.layers.Layer):
        raise ValueError()

    dest = growth_info.dest
    dest_layer = _as_same_type(src_layer, dest.layer)
    dest_keras_layer = _as_same_type(src_keras_layer, dest.keras_layer)

    return (
        src_keras_layer,
        src_keras_layer.get_weights(),
        dest_layer,
        dest_keras_layer,
        dest_keras_layer.get_weights(),
    )
