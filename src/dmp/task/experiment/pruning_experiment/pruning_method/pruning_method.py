from abc import ABC, abstractmethod
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


class PruningMethod(ABC):
    """
    Prunes a layer and its descendants in some way.
    """

    @abstractmethod
    def prune(
        self,
        root: Layer,
    ) -> int:  # returns number of weights pruned
        pass

    def is_prunable(
        self,
        layer: Layer,
    ) -> bool:
        keras_layer = layer.keras_layer
        return hasattr(keras_layer, "kernel_constraint") and isinstance(
            keras_layer.kernel_constraint, ParameterMask
        )

    def get_weights_and_mask(
        self,
        layer: Layer,
    ) -> Tuple[numpy.ndarray, tensorflow.Variable]:
        keras_layer = layer.keras_layer
        weights = keras_layer.get_weights()[0]  # type: ignore
        return (
            weights,  # type: ignore
            keras_layer.kernel_constraint.get_mask(weights.shape),  # type: ignore
        )

    def get_prunable_weights_from_layer(
        self,
        layer: Layer,
    ) -> numpy.ndarray:
        weights, mask = self.get_weights_and_mask(layer)
        return weights[mask].flatten()  # type: ignore

    def get_prunable_layers(
        self,
        root: Layer,
    ) -> List[Layer]:
        return [layer for layer in root.layers_post_ordered if self.is_prunable(layer)]

    def get_pruning_weights(self, prunable_layers: List[Layer]):
        return numpy.concatenate(
            [
                l
                for l in (
                    self.get_prunable_weights_from_layer(layer)
                    for layer in prunable_layers
                )
                if l is not None
            ]
        )

    def get_prunable_layers_and_weights(
        self,
        root: Layer,
    ) -> Tuple[List[Layer], numpy.ndarray]:
        prunable_layers = self.get_prunable_layers(root)
        prunable_weights = self.get_pruning_weights(prunable_layers)
        return prunable_layers, prunable_weights

    def prune_layers_using_mask(
        self,
        prunable_layers: List[Layer],
        prune_mask: numpy.ndarray,  # True at indicies of unpruned weights to prune, False otherwise
    ) -> int:
        # prune at selected level
        total_pruned = 0
        index = 0
        for layer in prunable_layers:

            # get layer mask
            weights, mask = self.get_weights_and_mask(layer)

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
