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


@dataclass
class WeightRenormalizingPruner(GlobalValuePruningMethod):
    """
    Pruning method that renormalizes weights before pruning
    """

    delegate: GlobalValuePruningMethod

    def compute_pruning_values(
        self,
        root: Layer,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
    ) -> numpy.ndarray:
        

        prunable_weights = numpy.hstack(
            [
                visitor.normalized_weight_map[layer].flatten()
                for layer in prunable_layers
            ]
        )

        return self.delegate.compute_pruning_values(
            root,
            prunable_layers,
            prunable_weights,
        )
