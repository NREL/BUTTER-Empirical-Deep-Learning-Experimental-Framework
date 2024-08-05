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


class WeightNormalizer(ABC):

    def compute_normalized_weights(
        self,
        root: Layer,
    ) -> Dict[Layer, numpy.ndarray]:

        class LayerVisitor:
            def __init__(self, parent: "WeightNormalizer", layer: Layer) -> None:
                self.parent = parent
                self.input_scale_map = {}
                self.normalized_weight_map = {}

                for layer in root.layers_post_ordered:
                    self.visit(layer)

            @singledispatchmethod
            def visit(self, target: Layer) -> None:
                self.passthrough_layer(target)

            @visit.register
            def _(self, target: PoolingLayer) -> None:
                self.passthrough_layer(target)

            @visit.register
            def _(self, target: Dense) -> None:
                self.normalize_layer(target)

            @visit.register
            def _(self, target: DenseConv) -> None:
                self.normalize_layer(target)

            @visit.register
            def _(self, target: SeparableConv) -> None:
                self.normalize_layer(target)

            def passthrough_layer(self, target: Layer) -> None:
                self.input_scale_map[target] = self.input_scale_map[target.input]

            def normalize_layer(self, target: Layer) -> None:
                keras_layer = target.keras_layer

                input_scale = numpy.hstack(
                    [
                        self.input_scale_map[input]
                        * numpy.ones(input.computed_shape[1:])
                        for input in target.inputs
                    ]
                )

                parameters = keras_layer.get_weights()  # type: ignore
                weights = parameters[0]

                input_scaled_weights = weights * input_scale
                input_norms = self.parent.compute_input_norms(input_scaled_weights)
                self.input_scale_map[target] = input_norms

                normalized_weights = (
                    input_scaled_weights / input_norms
                )  # .reshape(1,input_norms.size)

                self.normalized_weight_map[target] = normalized_weights

        # (input, input_scale)
        visitor = LayerVisitor(self, root)
        return visitor.normalized_weight_map

    @abstractmethod
    def compute_input_norms(self, input_scaled_weights):
        pass
