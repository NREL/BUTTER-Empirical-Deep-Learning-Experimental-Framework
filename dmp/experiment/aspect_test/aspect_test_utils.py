import gc
import json
import math
import os
import random
import sys
from copy import deepcopy
from functools import singledispatchmethod
from typing import Callable, Union, List, Optional

import numpy
import pandas
import tensorflow
from keras_buoy.models import ResumableModel
from sklearn.model_selection import train_test_split
from tensorflow.keras import (
    callbacks,
    metrics,
    optimizers,
)
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras import losses, Input
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

from command_line_tools import (
    command_line_config,
    run_tools,
)
from dmp.data.logging import write_log
from dmp.data.pmlb import pmlb_loader
from dmp.data.pmlb.pmlb_loader import load_dataset
from dmp.experiment.structure.algorithm.network_json_serializer import NetworkJSONSerializer
from dmp.experiment.structure.n_add import NAdd
from dmp.experiment.structure.n_dense import NDense
from dmp.experiment.structure.n_input import NInput
from dmp.experiment.structure.network_module import NetworkModule
from dmp.jq import jq_worker


def count_trainable_parameters_in_keras_model(model: Model) -> int:
    count = 0
    for var in model.trainable_variables:
        # print('ctp {}'.format(var.get_shape()))
        acc = 1
        for dim in var.get_shape():
            acc *= int(dim)
        # print('ctp acc {}'.format(acc))
        count += acc
    # print('ctp total {}'.format(count))
    return count


def count_num_free_parameters(target: NetworkModule) -> int:
    def build_set_of_modules(target: NetworkModule) -> set:
        cache = {target}
        cache.update(*map(build_set_of_modules, target.inputs))
        return cache

    class GetNumFreeParameters:
        """
        Visitor that counts the number of free (trainable) parameters in a NetworkModule
        """

        @singledispatchmethod
        def visit(self, target) -> any:
            raise Exception('Unsupported module of type "{}".'.format(type(target)))

        @visit.register
        def _(self, target: NInput) -> int:
            return 0

        @visit.register
        def _(self, target: NDense) -> int:
            return (sum((i.size for i in target.inputs)) + 1) * target.size

        @visit.register
        def _(self, target: NAdd) -> int:
            return 0

    return sum((GetNumFreeParameters().visit(i) for i in build_set_of_modules(target)))


def make_network(
        inputs: numpy.ndarray,
        widths: [int],
        residual_mode: any,
        input_activation,
        internal_activation,
        output_activation,
        depth,
        topology
) -> NetworkModule:
    # print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))

    input_layer = NInput(label=0,
                         inputs=[],
                         shape=list(inputs.shape[1:]))
    current = input_layer
    # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
    for d in range(depth):
        layer_width = widths[d]

        # Activation functions may be different for input, output, and hidden layers
        activation = internal_activation
        if d == 0:
            activation = input_activation
        elif d == depth - 1:
            activation = output_activation

        # Fully connected layer
        layer = NDense(label=0,
                       inputs=[current, ],
                       shape=[layer_width, ],
                       activation=activation)

        # Skip connections for residual modes
        assert residual_mode in ['full', 'none'], f"Invalid residual mode {residual_mode}"
        if residual_mode == 'full':
            assert topology in ['rectangle',
                                'wide_first'], f"Full residual mode is only compatible with rectangular and wide_first topologies, not {topology}"

            # in wide-first networks, first layer is of different dimension, and will therefore not originate any skip connections
            first_residual_layer = 1 if topology == "wide_first" else 0
            # Output layer is assumed to be of a different dimension, and will therefore not directly receive skip connections
            last_residual_layer = depth - 1
            if d > first_residual_layer and d < last_residual_layer:
                layer = NAdd(label=0,
                             inputs=[layer, current],
                             shape=layer.shape.copy())  ## TODO JP: Only works for rectangle

        current = layer

    return current


def build_keras_network(target: NetworkModule, model_cache: dict = {}) -> (tensorflow.keras.Model):
    """
    Recursively builds a keras network from the given network module and its directed acyclic graph of inputs
    :param target: starting point module
    :param model_cache: optional, used internally to preserve information through recursive calls
    :return: a list of input keras modules to the network and the output keras module
    """

    class MakeModelFromNetwork:
        """
        Visitor that makes a keras module for a given network module and input keras modules
        """

        @singledispatchmethod
        def visit(self, target, inputs: []) -> any:
            raise Exception('Unsupported module of type "{}".'.format(type(target)))

        @visit.register
        def _(self, target: NInput, inputs: []) -> any:
            return Input(shape=target.shape), True

        @visit.register
        def _(self, target: NDense, inputs: []) -> any:
            return Dense(
                target.shape[0],
                activation=target.activation,
            )(*inputs), False

        @visit.register
        def _(self, target: NAdd, inputs: []) -> any:
            return tensorflow.keras.layers.add(inputs), False

    network_inputs = {}
    keras_inputs = []

    for i in target.inputs:
        if i not in model_cache.keys():
            model_cache[i] = build_keras_network(i, model_cache)
        i_network_inputs, i_keras_module = model_cache[i]
        for new_input in i_network_inputs:
            network_inputs[new_input.ref()] = new_input
        keras_inputs.append(i_keras_module)

    keras_module, is_input = MakeModelFromNetwork().visit(target, keras_inputs)

    if is_input:
        network_inputs[keras_module.ref()] = keras_module

    keras_input = list(network_inputs.values())

    return Model(inputs=keras_input, outputs=keras_module)


def compute_network_configuration(num_outputs, dataset) -> (any, any):
    output_activation = 'relu'
    run_task = dataset['Task']
    if run_task == 'regression':
        run_loss = losses.mean_squared_error
        output_activation = 'sigmoid'
        # print('mean_squared_error')
    elif run_task == 'classification':
        if num_outputs == 1:
            output_activation = 'sigmoid'
            run_loss = losses.binary_crossentropy
            # print('binary_crossentropy')
        else:
            output_activation = 'softmax'
            run_loss = losses.categorical_crossentropy
            # print('categorical_crossentropy')
    else:
        raise Exception('Unknown task "{}"'.format(run_task))

    return output_activation, run_loss


def binary_search_int(objective: Callable[[int], Union[int, float]],
                      minimum: int,
                      maximum: int,
                      ) -> (int, bool):
    """
    :param objective: function for which to find fixed point
    :param minimum: min value of search
    :param maximum: max value of search
    :return: solution
    """
    if minimum > maximum:
        raise ValueError("binary search minimum must be less than maximum")
    candidate = 0
    while minimum < maximum:
        candidate = (maximum + minimum) // 2
        evaluation = objective(candidate)

        if evaluation < 0:  # candidate < target
            minimum = candidate + 1
        elif evaluation > 0:  # candidate > target
            maximum = candidate
        else:  # candidate == target
            return candidate, True
    return candidate, False


# def count_fully_connected_parameters(widths: [int]) -> int:
#     depth = len(widths)
#     p = (num_inputs + 1) * widths[0]
#     if depth > 1:
#         for k in range(1, depth):
#             p += (widths[k - 1] + 1) * widths[k]
#     return p


def find_best_layout_for_budget_and_depth(
        inputs,
        residual_mode,
        input_activation,
        internal_activation,
        output_activation,
        budget,
        make_widths: Callable[[int], List[int]],
        depth,
        topology
) -> (int, [int], NetworkModule):
    best = (math.inf, None, None)

    def search_objective(w0):
        nonlocal best
        widths = make_widths(w0)
        network = make_network(inputs,
                               widths,
                               residual_mode,
                               input_activation,
                               internal_activation,
                               output_activation,
                               depth,
                               topology
                               )
        delta = count_num_free_parameters(network) - budget

        if abs(delta) < abs(best[0]):
            best = (delta, widths, network)

        return delta

    binary_search_int(search_objective, 1, int(2 ** 30))
    return best


def get_rectangular_widths(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.extend((int(round(w0)) for k in range(0, depth - 1)))
        layout.append(num_outputs)
        return layout

    return make_layout


def get_trapezoidal_widths(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    def make_layout(w0):
        beta = (w0 - num_outputs) / (depth - 1)
        return [int(round(w0 - beta * k)) for k in range(0, depth)]

    return make_layout


def get_exponential_widths(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    def make_layout(w0):
        beta = math.exp(math.log(num_outputs / w0) / (depth - 1))
        return [max(num_outputs, int(round(w0 * (beta ** k)))) for k in range(0, depth)]

    return make_layout


def get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs: int, depth: int, first_layer_width_multiplier: float = 10,
) -> Callable[[float], List[int]]:
    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.append(w0)
        if depth > 2:
            inner_width = max(1, int(round(w0 / first_layer_width_multiplier)))
            layout.extend((inner_width for k in range(0, depth - 2)))
        layout.append(num_outputs)
        return layout

    return make_layout


def get_wide_first_5x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 5)


def get_wide_first_20x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 20)


def widths_factory(topology):
    if topology == 'rectangle':
        return get_rectangular_widths
    elif topology == 'trapezoid':
        return get_trapezoidal_widths
    elif topology == 'exponential':
        return get_exponential_widths
    elif topology == 'wide_first':
        return get_wide_first_layer_rectangular_other_layers_widths
    elif topology == 'wide_first_5x':
        return get_wide_first_5x
    elif topology == 'wide_first_20x':
        return get_wide_first_20x
    else:
        assert False, 'Topology "{}" not recognized.'.format(topology)

