from copy import copy
from dataclasses import dataclass
from functools import singledispatchmethod
from math import ceil
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy

from dmp import common

from dmp.layer import *
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.pruning_experiment.weight_mask import WeightMask
import dmp.keras_interface.keras_keys as keras_keys


class AccessModelWeights:
    @staticmethod
    def get_weights(
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        use_mask : bool = True,
    ) -> Dict[Layer, Any]:
        weight_map = {}

        def visit_weights(layer, keras_layer, layer_weights):
            weight_map[layer] = layer_weights
        
        mask_visitor = lambda keras_layer, layer_weights : None
        if use_mask:
            mask_visitor = lambda keras_layer, layer_weights: AccessModelWeights._visit_masks(
                AccessModelWeights._get_and_merge_mask, keras_layer, layer_weights)

        AccessModelWeights._visit_weights(
            root,
            lambda layer: layer_to_keras_map[layer].keras_layer,
            lambda layer, keras_layer: keras_layer.get_weights(),
            mask_visitor,
            visit_weights,
        )
        return weight_map

    @staticmethod
    def set_weights(
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        weight_map: Dict[Layer, Any],
        use_mask : bool = True,
    ) -> None:
        
        mask_visitor = lambda keras_layer, layer_weights : None
        if use_mask:
            mask_visitor = lambda keras_layer, layer_weights: AccessModelWeights._visit_masks(
                AccessModelWeights._set_mask, keras_layer, layer_weights)
            
        AccessModelWeights._visit_weights(
            root,
            lambda layer: AccessModelWeights._get_keras_layer_to_set(
                layer_to_keras_map, layer
            ),
            lambda layer, keras_layer: weight_map.get(layer, None),
            mask_visitor,
            lambda layer, keras_layer, layer_weights: keras_layer.set_weights(layer_weights),  # type: ignore
        )

    @staticmethod
    def _visit_weights(
        root: Layer,
        get_keras_layer,
        get_layer_weights,
        visit_masks,
        visit_weights,
    ) -> None:
        for layer in root.all_descendants:
            keras_layer = get_keras_layer(layer)
            if keras_layer is None or not AccessModelWeights._has_weights(keras_layer):
                continue
            layer_weights = get_layer_weights(layer, keras_layer)
            if layer_weights is None:
                continue
            visit_masks(keras_layer, layer_weights)
            visit_weights(layer, keras_layer, layer_weights)

    @staticmethod
    def _has_weights(keras_layer) -> bool:
        return hasattr(keras_layer, keras_keys.get_weights) and hasattr(
            keras_layer, keras_keys.set_weights
        )

    @staticmethod
    def _get_mask_constraint(keras_layer, constraint_member_name: str) -> Optional[Any]:
        if hasattr(keras_layer, constraint_member_name):
            constraint = getattr(keras_layer, constraint_member_name)
            if isinstance(constraint, WeightMask):
                return constraint
        return None

    @staticmethod
    def _get_keras_layer_to_set(
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        layer: Layer,
    ) -> Optional[Any]:
        layer_info = layer_to_keras_map.get(layer, None)
        if layer_info is None:
            return None
        return layer_info.keras_layer

    @staticmethod
    def _ignore_masks(
        keras_layer,
        layer_weights,
    ):
        pass

    @staticmethod
    def _visit_masks(
        visit_mask,
        keras_layer,
        layer_weights,
    ):
        layer_weights[0] = visit_mask(
            keras_layer,
            layer_weights[0],
            keras_keys.kernel_constraint,
        )

        if len(layer_weights) > 1:
            layer_weights[1] = visit_mask(
                keras_layer,
                layer_weights[1],
                keras_keys.bias_constraint,
            )

    @staticmethod
    def _get_and_merge_mask(keras_layer, weights, constraint_member_name: str):
        # convert masked weights to nan
        constraint = AccessModelWeights._get_mask_constraint(
            keras_layer,
            constraint_member_name,
        )
        if constraint is not None:
            weights = numpy.where(constraint.mask, weights, numpy.nan)
            # special_value = numpy.finfo(weights.dtype).smallest_subnormal
            # weights = numpy.where(weights == special_value, 0.0, weights)
            # weights = numpy.where(constraint.mask, weights, special_value)
        return weights

    @staticmethod
    def _set_mask(keras_layer, weights, constraint_member_name: str):
        # convert masked weights to nan
        constraint = AccessModelWeights._get_mask_constraint(
            keras_layer,
            constraint_member_name,
        )
        if constraint is not None:
            mask = numpy.logical_not(numpy.isnan(weights))
            # special_value = numpy.finfo(weights.dtype).smallest_subnormal
            # mask = numpy.logical_not(weights == special_value)
            constraint.mask = mask
            weights = numpy.where(mask, weights, 0.0)
        return weights
