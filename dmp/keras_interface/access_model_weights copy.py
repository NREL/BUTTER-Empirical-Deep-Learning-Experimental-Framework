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

from dmp.layer import *
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask



def get_weights(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    use_mask: bool = True,
) -> Dict[Layer, List[numpy.ndarray]]:
    weight_map: Dict[Layer, List[numpy.ndarray]] = {}

    mask_visitor = lambda keras_layer, layer_weights: None
    if use_mask:
        mask_visitor = lambda keras_layer, layer_weights: visit_masks(
            get_and_merge_mask,
            keras_layer,
            layer_weights,
        )

    visit_weights(
        root,
        lambda layer: layer_to_keras_map[layer].keras_layer,
        lambda layer, keras_layer: keras_layer.get_weights(),
        mask_visitor,
        lambda layer, keras_layer, layer_weights : weight_map.update({layer: layer_weights}),
    )
    return weight_map

def set_weights(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    weight_map: Dict[Layer, List[numpy.ndarray]],
    use_mask: bool = True,
) -> None:
    mask_visitor = lambda keras_layer, layer_weights: None
    if use_mask:
        mask_visitor = lambda keras_layer, layer_weights: visit_masks(
            set_mask,
            keras_layer,
            layer_weights,
        )

    visit_weights(
        root,
        lambda layer: _get_keras_layer_to_set(layer_to_keras_map, layer),
        lambda layer, keras_layer: weight_map.get(layer, None),
        mask_visitor,
        lambda layer, keras_layer, layer_weights: keras_layer.set_weights(layer_weights),  # type: ignore
    )

def visit_weights(
    root: Layer,
    get_keras_layer: Callable,
    get_layer_weights: Callable,
    visit_masks: Callable,
    visit_weights: Callable,
) -> None:
    for layer in root.layers:
        keras_layer = get_keras_layer(layer)
        if keras_layer is None or not _has_weights(keras_layer):
            continue
        layer_weights = get_layer_weights(layer, keras_layer)
        if layer_weights is None:
            continue
        visit_masks(keras_layer, layer_weights)
        visit_weights(layer, keras_layer, layer_weights)

def _has_weights( keras_layer) -> bool:
    return hasattr(keras_layer, 'get_weights') and hasattr(
        keras_layer, 'set_weights'
    )

def _get_keras_layer_to_set(
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    layer: Layer,
) -> Optional[Any]:
    layer_info = layer_to_keras_map.get(layer, None)
    if layer_info is None:
        return None
    return layer_info.keras_layer

def visit_masks(
    visit_mask,
    keras_layer,
    layer_weights,
):
    layer_weights[0] = visit_mask(
        keras_layer,
        layer_weights[0],
        'kernel_constraint',
    )

    if len(layer_weights) > 1:
        layer_weights[1] = visit_mask(
            keras_layer,
            layer_weights[1],
            'bias_constraint',
        )

def _get_mask_constraint(
    keras_layer,
    constraint_member_name: str,
) -> Optional[Any]:
    if hasattr(keras_layer, constraint_member_name):
        constraint = getattr(keras_layer, constraint_member_name)
        if isinstance(constraint, ParameterMask):
            return constraint
    return None

def get_and_merge_mask(
    keras_layer,
    weights,
    constraint_member_name: str,
):
    # convert masked weights to nan
    constraint = _get_mask_constraint(
        keras_layer,
        constraint_member_name,
    )
    if constraint is not None:
        weights = numpy.where(constraint.mask, weights, numpy.nan)
    return weights

def set_mask(cls, keras_layer, weights, constraint_member_name: str):
    # convert masked weights to nan
    constraint = _get_mask_constraint(
        keras_layer,
        constraint_member_name,
    )
    if constraint is not None:
        mask = numpy.logical_not(numpy.isnan(weights))
        constraint.mask = mask
        weights = numpy.where(mask, weights, 0.0)
    return weights

def lin_iterp_weights(
    weights_a: Dict[Layer, List[numpy.ndarray]],
    alpha: float,
    weights_b: Dict[Layer, List[numpy.ndarray]],
) -> Dict[Layer, List[numpy.ndarray]]:
    results = {}
    for layer, weights in weights_a.items():
        results[layer] = [
            weight_a * alpha + weight_b * (1.0 - alpha)
            for weight_a, weight_b in zip(weights, weights_b[layer])
        ]
    return results
