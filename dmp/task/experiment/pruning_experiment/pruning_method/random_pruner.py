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
from dmp.task.experiment.pruning_experiment.pruning_method.global_value_pruning_method import (
    GlobalValuePruningMethod,
)
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask


@dataclass
class RandomPruner(GlobalValuePruningMethod):
    def compute_pruning_values(
        self,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
    ) -> numpy.ndarray:
        return numpy.random.random(prunable_weights.shape)
