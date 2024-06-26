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


@dataclass
class ValuePruningMethod(PruningMethod):
    """
    Pruning method that prunes some proportion of the lowest-valued weights based on the "pruning value" assigned to them via the compute_pruning_values() method.
    """

    def prune_target_layers(
        self,
        root: Layer,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
        pruning_proportion: float,
    ):
        # set pruning masks to False where weights were pruned
        return self.prune_layers_using_mask(
            prunable_layers,
            self.make_pruning_mask(
                self.compute_pruning_values(
                    root,
                    prunable_layers,
                    prunable_weights,
                ),
                pruning_proportion,
            ),
        )

    def make_pruning_mask(
        self,
        weight_values: numpy.ndarray,
        pruning_proportion: float,
    ):

        # weight_index is an array of the indicies that would sort the prunbable weight values (ascending order)
        weight_index = numpy.argsort(weight_values)

        # create a pruning mask with the same shape as the weight_index
        prune_mask = numpy.zeros(weight_values.shape, dtype=bool)
        del weight_values

        # compute number to prune based on pruning rate
        how_many_to_prune = int(numpy.round(pruning_proportion * weight_index.size))

        # set the pruning mask indicies of the lowest valued weights to True
        prune_mask[weight_index[0:how_many_to_prune]] = True
        del weight_index

        return prune_mask

    @abstractmethod
    def compute_pruning_values(
        self,
        root: Layer,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
    ) -> numpy.ndarray:
        pass
