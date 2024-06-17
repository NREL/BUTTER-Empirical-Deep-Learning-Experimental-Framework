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
class GlobalValuePruningMethod(PruningMethod):
    """
    Pruning method that prunes some proportion of the lowest-valued weights based on the "pruning value" assigned to them via the compute_pruning_values() method.
    """

    pruning_rate: float  # proportion of additional weights to prune each time. (e.x. 0.10 means to prune 10% of the remaining weights on each call to prune())

    def prune_all_layers_using_mask(
        self,
        prunable_layers: List[Layer],
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        prune_mask: numpy.ndarray,  # True at indicies of unpruned weights to prune, False otherwise
    ) -> int:
        # prune at selected level
        total_pruned = 0
        index = 0
        for layer in prunable_layers:

            # get layer mask
            weights, mask = self.get_weights_and_mask(layer, layer_to_keras_map)

            # get mask as boolean numpy array
            mask_array = mask.numpy().astype(bool)

            # array of True's the shape/size of unpruned weights in this layer
            candidates = mask_array[mask_array]

            if len(mask_array.shape) == 0:
                mask_array = numpy.ndarray([int(mask_array)], dtype=bool)

            # make slice of prune mask for this layer's unpruned weights
            next_index = index + candidates.size
            layer_prune = (prune_mask[index:next_index]).reshape(candidates.shape)
            index = next_index

            # set candidates mask to False where layer_prune / prune mask is True
            candidates[layer_prune] = False

            # set mask_array to have candidate's values over the previously unpruned weights
            mask_array[mask_array] = candidates

            total_pruned += mask_array.size - mask_array.sum()

            # set mask to new mask_array value
            mask.assign(mask_array)

        return total_pruned

    def prune(
        self,
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> int:  # returns number of weights pruned
        prunable_layers, prunable_weights = self.get_prunable_layers_and_weights(
            root, layer_to_keras_map
        )

        prunable_weights = self.compute_pruning_values(
            prunable_layers,
            prunable_weights,
        )

        # weight_index is an array of the indicies that would sort the prunbable weight values (ascending order)
        weight_index = numpy.argsort(prunable_weights)

        # create a pruning mask with the same shape as the weight_index
        prune_mask = numpy.zeros(prunable_weights.shape, dtype=bool)
        del prunable_weights

        # compute number to prune based on pruning rate
        how_many_to_prune = int(numpy.round(self.pruning_rate * weight_index.size))

        # set the pruning mask indicies of the lowest valued weights to True
        prune_mask[weight_index[0:how_many_to_prune]] = True
        del weight_index

        # set pruning masks to False where weights were pruned
        total_pruned = self.prune_all_layers_using_mask(
            prunable_layers,
            layer_to_keras_map,
            prune_mask,
        )

        return total_pruned

    @abstractmethod
    def compute_pruning_values(
        self,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
    ) -> numpy.ndarray:
        pass
