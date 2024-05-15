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
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> int:  # returns number of weights pruned
        pass

    def is_prunable(
        self,
        layer: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> bool:
        keras_layer = layer_to_keras_map[layer].keras_layer
        return hasattr(keras_layer, "kernel_constraint") and isinstance(
            keras_layer.kernel_constraint, ParameterMask
        )

    def get_weights_and_mask(
        self,
        layer: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> Tuple[numpy.ndarray, tensorflow.Variable]:
        keras_layer = layer_to_keras_map[layer].keras_layer
        weights = keras_layer.get_weights()[0]  # type: ignore
        return (
            weights,  # type: ignore
            keras_layer.kernel_constraint.get_mask(weights.shape),  # type: ignore
        )

    def get_prunable_weights_from_layer(
        self,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        layer: Layer,
    ) -> numpy.ndarray:
        weights, mask = self.get_weights_and_mask(layer, layer_to_keras_map)
        return weights[mask].flatten()  # type: ignore

    def get_prunable_layers(
        self,
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> List[Layer]:
        return [
            layer
            for layer in root.layers
            if self.is_prunable(layer, layer_to_keras_map)
        ]

    def get_prunable_layers_and_weights(
        self,
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> Tuple[List[Layer], numpy.ndarray]:
        prunable_layers = self.get_prunable_layers(root, layer_to_keras_map)
        prunable_weights = numpy.concatenate(
            [
                l
                for l in (
                    self.get_prunable_weights_from_layer(layer_to_keras_map, layer)
                    for layer in prunable_layers
                )
                if l is not None
            ]
        )
        return prunable_layers, prunable_weights
