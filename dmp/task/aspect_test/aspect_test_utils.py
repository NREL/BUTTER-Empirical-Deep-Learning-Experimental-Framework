from dataclasses import dataclass
import math
import random
from copy import deepcopy
from functools import singledispatchmethod
import time
from typing import Callable, Dict, Tuple, Union, List, Optional

import numpy
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import losses, Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
import tensorflow.keras as keras

from dmp.structure.n_add import NAdd
from dmp.structure.n_dense import NDense
from dmp.structure.n_input import NInput
from dmp.structure.network_module import NetworkModule


def set_random_seeds(seed: Optional[int]) -> int:
    if seed is None:
        seed = time.time_ns()

    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    random.seed(seed)
    return seed


def add_label_noise(label_noise, run_task, train_outputs):
    if label_noise is not None and label_noise != 'none':
        train_size = len(train_outputs)
        # print(f'run_task {run_task} output shape {outputs.shape}')
        # print(f'sample\n{outputs_train[0:20, :]}')
        if run_task == 'classification':
            num_to_perturb = int(train_size * label_noise)
            noisy_labels_idx = numpy.random.choice(
                train_size, size=num_to_perturb, replace=False)

            num_outputs = train_outputs.shape[1]
            if num_outputs == 1:
                # binary response variable...
                train_outputs[noisy_labels_idx] ^= 1
            else:
                # one-hot response variable...
                rolls = numpy.random.choice(numpy.arange(
                    num_outputs - 1) + 1, noisy_labels_idx.size)
                for i, idx in enumerate(noisy_labels_idx):
                    train_outputs[noisy_labels_idx] = numpy.roll(
                        train_outputs[noisy_labels_idx], rolls[i])
                # noisy_labels_new_idx = numpy.random.choice(train_size, size=num_to_perturb, replace=True)
                # outputs_train[noisy_labels_idx] = outputs_train[noisy_labels_new_idx]
        elif run_task == 'regression':
            # mean = numpy.mean(outputs, axis=0)
            std_dev = numpy.std(train_outputs, axis=0)
            # print(f'std_dev {std_dev}')
            noise_std = std_dev * label_noise
            for i in range(train_outputs.shape[1]):
                train_outputs[:, i] += numpy.random.normal(
                    loc=0, scale=noise_std[i], size=train_outputs[:, i].shape)
        else:
            raise ValueError(
                f'Do not know how to add label noise to dataset task {run_task}.')


def prepare_dataset(
    test_split_method: str,
    split_portion: float,
    label_noise: float,
    run_config: dict,
    run_task: str,
    inputs,
    outputs,
) -> Dict[str, any]:
    run_config = deepcopy(run_config)
    if test_split_method == 'shuffled_train_test_split':

        train_inputs, test_inputs, train_outputs, test_outputs = \
            train_test_split(
                inputs,
                outputs,
                test_size=split_portion,
                shuffle=True,
            )
        add_label_noise(label_noise, run_task, train_outputs)

        run_config['validation_data'] = (test_inputs, test_outputs)
        run_config['x'] = train_inputs
        run_config['y'] = train_outputs
    else:
        raise NotImplementedError(
            f'Unknown test_split_method {test_split_method}.')
        # run_config['x'] = inputs
        # run_config['y'] = outputs
    return run_config


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
            raise Exception(
                'Unsupported module of type "{}".'.format(type(target)))

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
        widths: List[int],
        residual_mode: any,
        input_activation: str,
        internal_activation: str,
        output_activation: str,
        depth: int,
        shape: str,
        layer_args: dict,
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
                       activation=activation,
                       **layer_args
                       )

        # Skip connections for residual modes
        if residual_mode is None or residual_mode == 'none':
            pass
        elif residual_mode == 'full':
            # in wide-first networks, first layer is of different dimension, and will therefore not originate any skip connections
            first_residual_layer = 1 if shape == "wide_first" else 0
            # Output layer is assumed to be of a different dimension, and will therefore not directly receive skip connections
            last_residual_layer = depth - 1
            if d > first_residual_layer and d < last_residual_layer:
                layer = NAdd(label=0,
                             inputs=[layer, current],
                             shape=layer.shape.copy())  # TODO JP: Only works for rectangle
        else:
            raise NotImplementedError(
                f'Unknown residual mode "{residual_mode}".')

        current = layer

    return current


def make_regularizer(regularization_settings: dict) \
        -> keras.regularizers.Regularizer:
    if regularization_settings is None:
        return None
    name = regularization_settings['type']
    args = regularization_settings.copy()
    del args['type']

    cls = None
    if name == 'l1':
        cls = keras.regularizers.L1
    elif name == 'l2':
        cls = keras.regularizers.L2
    elif name == 'l1l2':
        cls = keras.regularizers.L1L2
    else:
        raise NotImplementedError(f'Unknown regularizer "{name}".')

    return cls(**args)


class MakeKerasLayersFromNetwork:
    """
    Visitor that makes a keras module for a given network module and input keras modules
    """
    _inputs: list = []
    _nodes: dict = {}
    _output: any = None

    def __init__(self, target: NetworkModule) -> None:
        self._output = self._visit(target)

    def __call__(self) -> Tuple[list, any]:
        return self._inputs, self._output

    def _visit(self, target: NetworkModule) -> any:
        if target not in self._nodes:
            keras_inputs = [self._visit(i) for i in target.inputs]
            self._nodes[target] = self._visit_raw(target, keras_inputs)
        return self._nodes[target]

    @singledispatchmethod
    def _visit_raw(self, target, keras_inputs) -> any:
        raise NotImplementedError(
            'Unsupported module of type "{}".'.format(type(target)))

    @_visit_raw.register
    def _(self, target: NInput, keras_inputs) -> any:
        result = Input(shape=target.shape)
        self._inputs.append(result)
        return result

    @_visit_raw.register
    def _(self, target: NDense, keras_inputs) -> any:

        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        return Dense(
            target.shape[0],
            activation=target.activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NAdd, keras_inputs) -> any:
        return tensorflow.keras.layers.add(keras_inputs)


def make_keras_network_from_network_module(target: NetworkModule, model_cache: dict = {}):
    """
    Recursively builds a keras network from the given network module and its directed acyclic graph of inputs
    :param target: starting point module
    :param model_cache: optional, used internally to preserve information through recursive calls
    """
    inputs, output = MakeKerasLayersFromNetwork(target)()
    return Model(inputs=inputs, outputs=[output])


def compute_network_configuration(num_outputs, dataset) -> Tuple[any, any]:
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
                      ) -> Tuple[int, bool]:
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
        size,
        make_widths: Callable[[int], List[int]],
        depth,
        shape,
        layer_args
) -> Tuple[int, List[int], NetworkModule]:
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
                               shape,
                               layer_args
                               )
        delta = count_num_free_parameters(network) - size

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


def get_wide_first_2x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 2)


def get_wide_first_4x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 2)


def get_wide_first_8x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 2)


def get_wide_first_16x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 2)


def get_wide_first_5x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 5)


def get_wide_first_20x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 20)


def widths_factory(shape):
    if shape == 'rectangle':
        return get_rectangular_widths
    elif shape == 'trapezoid':
        return get_trapezoidal_widths
    elif shape == 'exponential':
        return get_exponential_widths
    elif shape == 'wide_first_2x':
        return get_wide_first_2x
    elif shape == 'wide_first_4x':
        return get_wide_first_4x
    elif shape == 'wide_first_5x':
        return get_wide_first_5x
    elif shape in {'wide_first', 'wide_first_10x'}:
        return get_wide_first_layer_rectangular_other_layers_widths
    elif shape == 'wide_first_16x':
        return get_wide_first_16x
    elif shape == 'wide_first_20x':
        return get_wide_first_20x
    else:
        assert False, 'Shape "{}" not recognized.'.format(shape)
