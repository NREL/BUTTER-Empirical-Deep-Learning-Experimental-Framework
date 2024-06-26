from abc import ABC, abstractmethod
from dataclasses import dataclass
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

from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.parameter_mask import ParameterMask
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_method.value_pruning_method import (
    ValuePruningMethod,
)


@dataclass
class GlobalValuePruningMethod(ValuePruningMethod):
    """
    Pruning method that prunes some proportion of the lowest-valued weights based on the "pruning value" assigned to them via the compute_pruning_values() method.
    """

    pruning_rate: float  # proportion of additional weights to prune each time. (e.x. 0.10 means to prune 10% of the remaining weights on each call to prune())

    def prune(
        self,
        root: Layer,
    ) -> int:  # returns number of weights pruned
        prunable_layers, prunable_weights = self.get_prunable_layers_and_weights(
            root,
        )

        return self.prune_target_layers(
            root,
            prunable_layers,
            prunable_weights,
            self.pruning_rate,
        )
