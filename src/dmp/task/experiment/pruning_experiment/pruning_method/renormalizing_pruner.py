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
from dmp.task.experiment.pruning_experiment.pruning_method.global_value_pruning_method import (
    GlobalValuePruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)


@dataclass
class RenormalizingPruner(GlobalValuePruningMethod):
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

        input_scale_map = {}
        normalized_weight_map = {}
        # (input, input_scale)

        def scale_layer(layer: Layer):
            keras_layer = layer.keras_layer

            input_scale = numpy.hstack(
                [
                    input_scale_map[input] * numpy.ones(input.computed_shape[1:])
                    for input in layer.inputs
                ]
            )

            parameters = keras_layer.get_weights()  # type: ignore
            weights = parameters[0]

            input_scaled_weights = weights * input_scale
            input_norms = (numpy.square(input_scaled_weights)).sum(
                axis=1
            )  # sum along rows
            input_scale_map[layer] = input_norms

            normalized_weights = (
                input_scaled_weights / input_norms
            )  # .reshape(1,input_norms.size)

            normalized_weight_map[layer] = normalized_weights

        for layer in root.layers_post_ordered:
            scale_layer(layer)

        prunable_weights = numpy.hstack(
            [normalized_weight_map[layer].flatten() for layer in prunable_layers]
        )

        return self.delegate.compute_pruning_values(
            root,
            prunable_layers,
            prunable_weights,
        )
