from functools import singledispatchmethod
from typing import Any, Dict, Iterable, Set, Tuple

import numpy
import tensorflow
import tensorflow.keras.layers as layers
from dmp.layer.layer import *
from dmp.layer.visitor.keras_interface.layer_to_keras import KerasLayer


class OverlayGrowthMethod:
    """
    Visitor that fills one network with the values from another. 
    If the destination network is larger, this will 'grow' the source into
    the destination.

    Old to old weights are retained.
    Old to new weights are left as-is
    New to new weights are left as-is
    New to old weights are scaled by the 'scale' parameter
    """

    def __init__(
        self,
        source: Layer,
        source_layer_to_keras_map: Dict[Layer, Tuple[KerasLayer,
                                                     tensorflow.Tensor]],
        dest: Layer,
        dest_layer_to_keras_map: Dict[Layer, Tuple[KerasLayer,
                                                   tensorflow.Tensor]],
        old_to_old_scale: float = 1.0,
        new_to_new_scale: float = 1.0,
        old_to_new_scale: float = 1.0,
        new_to_old_scale: float = 0.0,
        new_add_to_old_scale: float = 0.0,
    ) -> None:
        self._source: Layer = source
        self._source_layer_to_keras_map: Dict[Layer, Tuple[
            KerasLayer, tensorflow.Tensor]] = source_layer_to_keras_map
        self._dest: Layer = dest
        self._dest_layer_to_keras_map: Dict[Layer, Tuple[
            KerasLayer, tensorflow.Tensor]] = dest_layer_to_keras_map

        self.old_to_old_scale: float = old_to_old_scale
        self.new_to_new_scale: float = new_to_new_scale
        self.old_to_new_scale: float = old_to_new_scale
        self.new_to_old_scale: float = new_to_old_scale
        self.new_add_to_old_scale: float = new_add_to_old_scale

        self._visited: Set[Layer] = set()

        self._visit(source, dest)

    # def __call__(self) -> Tuple[list, Any]:
    #     return self._inputs, self._outputs

    def _visit(
        self,
        source_layer: Layer,
        dest_layer: Layer,
    ) -> None:
        if source_layer in self._visited or dest_layer in self._visited:
            return
        self._visited.add(source_layer)
        self._visited.add(dest_layer)

        self._do_visit(source_layer, dest_layer)

        for s, d in zip(source_layer.inputs, dest_layer.inputs):
            self._visit(s, d)

    @singledispatchmethod
    def _do_visit(
        self,
        source_layer: Layer,
        dest_layer: Layer,
    ) -> None:
        raise NotImplementedError('Unsupported module of type "{}".'.format(
            type(source_layer)))

    @_do_visit.register
    def _(
        self,
        source_layer: Input,
        dest_layer: Layer,
    ) -> None:
        return

    @_do_visit.register
    def _(
        self,
        source_layer: Dense,
        dest_layer: Layer,
    ) -> None:
        self._check_same_type_layer(source_layer, dest_layer)

        source_keras: layers.Dense = \
            self._source_layer_layer_map[source_layer]  # type: ignore
        dest_keras: layers.Dense = \
            self._dest_layer_layer_map[dest_layer]  # type: ignore

        source_weights, source_biases = \
            source_keras.get_weights()  # type: ignore
        dest_weights, dest_biases = dest_keras.get_weights()  # type: ignore

        sw_shape = source_weights.shape
        sb_shape = source_biases.shape

        src_in_idx = sw_shape[0]
        src_out_idx = sw_shape[1]

        # scale old to new nodes
        self._scale_block(
            dest_weights[:src_in_idx, src_out_idx:],
            self.old_to_new_scale,
        )

        # scale new to old nodes
        self._scale_block(
            dest_weights[src_in_idx:, :src_out_idx],
            self.new_to_old_scale,
        )

        # scale new nodes
        self._scale_block(
            dest_weights[src_in_idx:, src_out_idx:],
            self.new_to_new_scale,
        )
        self._scale_block(
            dest_biases[sb_shape[0]:],
            self.new_to_new_scale,
        )

        # copy and scale old nodes
        self._blend_blocks(
            source_weights,
            self.new_add_to_old_scale,
            dest_weights[:src_in_idx, :src_out_idx],
            self.old_to_old_scale,
        )

        self._blend_blocks(
            source_biases,
            self.new_add_to_old_scale,
            dest_biases[:sb_shape[0]],
            self.old_to_old_scale,
        )

        dest_keras.set_weights(dest_weights)

    @_do_visit.register
    def _(
        self,
        source_layer: Add,
        dest_layer: Layer,
    ) -> None:
        return

    def _check_same_type_layer(
        self,
        source_layer: Layer,
        dest_layer: Layer,
    ) -> None:
        if type(source_layer) != type(dest_layer):
            raise TypeError(
                f'Source and destination Layer are not the same type {source_layer} != {dest_layer}.'
            )

    def flatten(self, items):
        """Yield items from any nested iterable; see Reference."""
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from self.flatten(x)
            else:
                yield x

    def _scale_block(
        self,
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
        self,
        source_weights: numpy.ndarray,
        source_scale: float,
        dest_block: numpy.ndarray,
        dest_scale: float,
    ) -> None:
        if source_scale == 0.0:
            dest_block[:] = source_weights
            self._scale_block(
                dest_block,
                source_scale,
            )
        else:
            self._scale_block(
                dest_block,
                dest_scale,
            )
            dest_block[:] += source_weights * source_scale
