from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
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
import tensorflow

from dmp.layer.batch_normalization import BatchNormalization
from dmp.layer.dense import Dense
from dmp.layer.dense_conv import DenseConv
from dmp.layer.layer import Layer
from dmp.layer.pooling_layer import PoolingLayer
from dmp.layer.separable_conv import SeparableConv
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask
from dmp.task.experiment.pruning_experiment.pruning_method.global_value_pruning_method import (
    GlobalValuePruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_method.weight_normalization.weight_normalizer import (
    WeightNormalizer,
)


class LInfinityWeightNormalizer(WeightNormalizer):

    def compute_input_norms(self, input_scaled_weights):
        return numpy.max(numpy.abs(input_scaled_weights), axis=1)  # sum along rows
