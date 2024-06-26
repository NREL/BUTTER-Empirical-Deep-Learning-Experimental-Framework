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

import numpy
from dmp.keras_interface.access_model_parameters import (
    get_mask_constraint,
    is_mask_constraint,
)
from dmp.layer import *

from dmp.layer.batch_normalization import BatchNormalization
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask
import tensorflow.keras as keras


def count_masked_parameters(
    target: Layer,
) -> int:
    num_masked_parameters = 0
    for layer in target.layers:
        keras_layer = layer.keras_layer
        if not isinstance(keras_layer, keras.layers.Layer):
            continue
        for variable in keras_layer.variables:  # type: ignore
            constraint = get_mask_constraint(keras_layer, variable)
            if constraint is None:
                continue
            numpy_mask = constraint.mask.numpy()
            num_masked_parameters += numpy_mask.size - numpy_mask.sum()
    return num_masked_parameters
