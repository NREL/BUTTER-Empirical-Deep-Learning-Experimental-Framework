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
from dmp.task.experiment.pruning_experiment.pruning_method.weight_normalization.weight_normalizer import WeightNormalizer


@dataclass
class WeightNormalizedMagnitudePruner(GlobalValuePruningMethod):

    weight_normalizer : WeightNormalizer

    def compute_pruning_values(
        self,
        root: Layer,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
    ) -> numpy.ndarray:
        normalized_weights = 

        return numpy.abs(prunable_weights)
