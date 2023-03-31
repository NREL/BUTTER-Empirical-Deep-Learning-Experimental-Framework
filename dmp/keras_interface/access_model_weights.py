from copy import copy
from dataclasses import dataclass
from functools import singledispatchmethod
from math import ceil
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union

import numpy

from dmp import common

from dmp.layer import *
from dmp.model.keras_layer_info import KerasLayerInfo
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import PruningMethod
from dmp.task.experiment.pruning_experiment.weight_mask import WeightMask


class AccessModelWeights():

    @staticmethod
    def get_weights(
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> Dict[Layer, Any]:
        weight_map = {}
        for layer in root.all_descendants:
            keras_layer = layer_to_keras_map[layer].keras_layer
            weight_map[layer] = keras_layer.get_weights()  # type: ignore

        return weight_map

    @staticmethod
    def set_weights(
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
        weight_map: Dict[Layer, Any],
    ) -> None:
        for layer in root.all_descendants:
            layer_weights = weight_map.get(layer, None)
            if layer_weights is None:
                continue

            keras_layer = layer_to_keras_map[layer].keras_layer
            keras_layer.set_weights(layer_weights)  # type: ignore
