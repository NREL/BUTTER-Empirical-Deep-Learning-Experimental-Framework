from functools import singledispatchmethod
import math
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
from dmp.keras_interface.access_model_parameters import get_mask_constraint
from dmp.layer import *

from dmp.layer.batch_normalization import BatchNormalization
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask


def count_masked_parameters(
    target: Layer,
    layer_to_keras_map: Dict[Layer, KerasLayerInfo],
) -> int:
    num_masked_parameters = 0
    for layer in target.layers:
        constraint = get_mask_constraint(
            layer,
            layer_to_keras_map.get(layer).keras_layer,  # type: ignore
        )
        if constraint is not None:
            print(
                f"found constraint with {constraint.mask.numpy().sum()} masked parameters"
            )
            num_masked_parameters += constraint.mask.numpy().sum()
    return num_masked_parameters
