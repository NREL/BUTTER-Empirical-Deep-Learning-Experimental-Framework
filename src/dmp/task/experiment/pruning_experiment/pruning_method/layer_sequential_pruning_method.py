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
class LayerSequentialPruningMethod(ValuePruningMethod):
    """ """

    pruning_rate: float  # proportion of additional weights to prune each time. (e.x. 0.10 means to prune 10% of the remaining weights on each call to prune())
    current_layer_index: int = -1

    def prune(
        self,
        root: Layer,
    ) -> int:  # returns number of weights pruned
        prunable_layers = self.get_prunable_layers(root)
        self.current_layer_index = (self.current_layer_index + 1) % len(prunable_layers)
        layer = prunable_layers[self.current_layer_index]

        return self.prune_target_layers(
            root,
            [layer],
            self.get_prunable_weights_from_layer(layer),
            self.pruning_rate,
        )
