from copy import copy
from functools import singledispatchmethod
from math import ceil
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union

import numpy

from dmp import common

from dmp.layer import *
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.lottery_ticket_experiment.weight_mask import WeightMask


class MagnitudePruner():

    kernel_constraint = 'kernel_constraint'
    prunable_types = {Dense, ConvolutionalLayer}

    def prune(
        self,
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        prune_percent: float,
    ) -> None:
        import tensorflow

        def is_prunable(
            layer: Layer,
        ) -> bool:
            if type(layer) not in self.prunable_types:
                return False
            keras_layer = layer_to_keras_map[layer]
            constraint = keras_layer.kernel_constraint  # type: ignore
            return isinstance(constraint, 'WeightMask')

        def get_weights_and_mask(
            layer: Layer, ) -> Tuple[numpy.ndarray, tensorflow.Variable]:
            keras_layer = layer_to_keras_map[layer]
            return (
                layer_to_keras_map[layer].get_weights()[0],  # type: ignore
                keras_layer.kernel_constraint.mask,  # type: ignore
            )

        def get_prunable_weights_from_layer(layer: Layer, ) -> numpy.ndarray:
            weights, mask = get_weights_and_mask(layer)
            return (weights[mask.numpy() != 0]).flatten()

        prunable_layers = [
            layer for layer in root.all_descendants if is_prunable(layer)
        ]

        prunable_weights = numpy.concatenate(
            list(
                filter(lambda w: w is not None,
                       (get_prunable_weights_from_layer(layer)
                        for layer in prunable_layers))))

        self.pruning_threshold = numpy.quantile(prunable_weights,
                                                prune_percent)
        del prunable_weights

        # prune at selected level
        for layer in prunable_layers:
            weights, mask = get_weights_and_mask(layer)
            mask.assign(mask.numpy() | (weights <= self.pruning_threshold))
