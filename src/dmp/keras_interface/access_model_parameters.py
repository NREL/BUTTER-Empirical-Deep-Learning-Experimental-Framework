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
import re
import numpy

from dmp.layer import *
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask
import tensorflow.keras as keras


def get_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    use_mask: bool,
) -> Dict[Layer, List[numpy.ndarray]]:
    parameter_map: Dict[Layer, List[numpy.ndarray]] = {}

    def visit_variable(layer, keras_layer, i, variable):
        value = variable.numpy()
        if use_mask:
            constraint = get_mask_constraint(keras_layer, variable)
            if constraint is not None:
                value = numpy.where(
                    constraint.mask.numpy(),
                    value,
                    numpy.nan,
                )
        parameter_map.setdefault(layer, []).append(value)

    visit_parameters(
        root,
        layer_to_keras_map,
        visit_variable,
    )
    return parameter_map


def set_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    parameter_map: Dict[Layer, List[numpy.ndarray]],
    restore_mask: bool,
) -> None:
    def visit_variable(layer, keras_layer, i, variable):
        value_list = parameter_map.get(layer, None)
        if value_list is not None:
            value = value_list[i]
            if restore_mask:
                constraint = get_mask_constraint(keras_layer, variable)
                if constraint is not None:
                    mask = numpy.logical_not(numpy.isnan(value))
                    constraint.mask = mask
                    value = numpy.where(mask, value, 0.0)
            variable.assign(value)

    visit_parameters(
        root,
        layer_to_keras_map,
        visit_variable,
    )


def visit_parameters(
    root: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    visit_variable: Callable,
) -> None:
    for layer in root.layers:
        layer_info = layer_to_keras_map.get(layer, None)
        if layer_info is None:
            continue

        keras_layer = layer_info.keras_layer
        if keras_layer is None or not isinstance(keras_layer, keras.layers.Layer):
            continue

        for i, variable in enumerate(keras_layer.variables):  # type: ignore
            visit_variable(layer, keras_layer, i, variable)


def get_mask_constraint(
    keras_layer,
    variable,
) -> Optional[Any]:
    match = re.fullmatch(r"^.*(bias|kernel)", variable.name)
    if match is not None:
        match_str = match.group(1)
        constraint_member_name = f"{match_str}_constraint"
        if hasattr(keras_layer, constraint_member_name):
            return is_mask_constraint(getattr(keras_layer, constraint_member_name))
    return None


def is_mask_constraint(
    constraint,
) -> Optional[Any]:
    if isinstance(constraint, ParameterMask):
        return constraint
    return None


def lin_iterp_parameters(
    parameters_a: Dict[Layer, List[numpy.ndarray]],
    alpha: float,
    parameters_b: Dict[Layer, List[numpy.ndarray]],
) -> Dict[Layer, List[numpy.ndarray]]:
    results = {}
    for layer, parameters in parameters_a.items():
        results[layer] = [
            parameter_a * alpha + parameter_b * (1.0 - alpha)
            for parameter_a, parameter_b in zip(parameters, parameters_b[layer])
        ]
    return results
