import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy
import tensorflow
import tensorflow.keras as keras
from dmp.structure.n_add import NAdd
from dmp.structure.n_dense import NDense
from dmp.structure.n_input import NInput
from dmp.structure.network_module import NetworkModule
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, losses
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from dmp.structure.n_conv import *
from dmp.structure.n_cell import *
from cnn.cell_structures import *


def add_label_noise(label_noise, run_task, train_outputs):
    if label_noise is not None and label_noise != 'none' and label_noise != 0.0:
        train_size = len(train_outputs)
        # print(f'run_task {run_task} output shape {outputs.shape}')
        # print(f'sample\n{outputs_train[0:20, :]}')
        if run_task == 'classification':
            num_to_perturb = int(train_size * label_noise)
            noisy_labels_idx = numpy.random.choice(train_size,
                                                   size=num_to_perturb,
                                                   replace=False)

            num_outputs = train_outputs.shape[1]
            if num_outputs == 1:
                # binary response variable...
                train_outputs[noisy_labels_idx] ^= 1
            else:
                # one-hot response variable...
                rolls = numpy.random.choice(
                    numpy.arange(num_outputs - 1) + 1, noisy_labels_idx.size)
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
                f'Do not know how to add label noise to dataset task {run_task}.'
            )


def prepare_dataset(
    test_split_method: str,
    split_portion: float,
    label_noise: float,
    run_config: dict,
    run_task: str,
    inputs,
    outputs,
    val_portion: Optional[float] = None,
) -> Dict[str, Any]:
    prepared_config = deepcopy(run_config)
    if test_split_method == 'shuffled_train_test_split':

        train_inputs, test_inputs, train_outputs, test_outputs = \
            train_test_split(
                inputs,
                outputs,
                test_size=split_portion,
                shuffle=True,
            )
        add_label_noise(label_noise, run_task, train_outputs)

        prepared_config['validation_data'] = (test_inputs, test_outputs)
        prepared_config['x'] = train_inputs
        prepared_config['y'] = train_outputs
    elif test_split_method == 'shuffled_train_val_test_split':

        train_inputs, test_inputs, train_outputs, test_outputs = \
            train_test_split(
                inputs,
                outputs,
                test_size=split_portion,
                shuffle=True,
            )
        if val_portion is None:
            val_portion = 0.0

        train_inputs, val_inputs, train_outputs, val_outputs = \
            train_test_split(
                train_inputs,
                train_outputs,
                test_size=int(val_portion/(1-split_portion)),
                shuffle=True,
            )
        add_label_noise(label_noise, run_task, train_outputs)

        prepared_config['test_data'] = (test_inputs, test_outputs)
        prepared_config['validation_data'] = (val_inputs, val_outputs)
        prepared_config['x'] = train_inputs
        prepared_config['y'] = train_outputs
    else:
        raise NotImplementedError(
            f'Unknown test_split_method {test_split_method}.')
        # run_config['x'] = inputs
        # run_config['y'] = outputs
    return prepared_config


def make_network_module_graph(
    input_shape: Tuple[int, ...],
    widths: List[int],
    residual_mode: Optional[str],
    input_activation: str,
    internal_activation: str,
    output_activation: str,
    layer_args: dict,
) -> NetworkModule:
    # print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))

    input_layer = NInput(shape=list(input_shape[1:]))
    current = input_layer
    # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
    for d, layer_width in enumerate(widths):

        # Activation functions may be different for input, output, and hidden layers
        activation = internal_activation
        if d == 0:
            activation = input_activation
        elif d == len(widths) - 1:
            activation = output_activation

        # Fully connected layer
        layer = NDense(
            inputs=[
                current,
            ],
            shape=[
                layer_width,
            ],
            activation=activation,
            **layer_args,
        )

        # Skip connections for residual modes
        if residual_mode is None or residual_mode == 'none':
            pass
        elif residual_mode == 'full':
            # If this isn't the first or last layer, and the previous layer is
            # of the same width insert a residual sum between layers
            # NB: Only works for rectangle
            if d > 0 and d < len(widths) - 1 and layer_width == widths[d - 1]:
                layer = NAdd(inputs=[layer, current])
        else:
            raise NotImplementedError(
                f'Unknown residual mode "{residual_mode}".')
        current = layer
    return current


def get_from_config_mapping(
    name: Any,
    mapping: Dict[Any, Any],
    config_name: str,
) -> Any:
    if name in mapping:
        return mapping[name]
    raise NotImplementedError(f'Unknown {config_name} "{name}".')


def get_params_and_type_from_config(
    config: dict,
    type_key: str = 'type',
) -> Tuple[str, dict]:
    params = config.copy()
    del params[type_key]
    return config[type_key], params


def make_from_typed_config(
    config: dict,
    mapping: Dict[str, Callable],
    config_name: str,
    *args,
    **kwargs,
) -> Any:
    type, params = get_params_and_type_from_config(config)
    factory = get_from_config_mapping(type, mapping, config_name)
    return factory(*args, **kwargs, **params)

def make_keras_regularizer(config: Optional[Dict]) \
        -> Optional[keras.regularizers.Regularizer]:
    if config is None:
        return None
    return make_from_typed_config(
        config, {
            'l1': keras.regularizers.L1,
            'l2': keras.regularizers.L2,
            'l1l2': keras.regularizers.L1L2
        }, 'regularizer')


def get_activation_factory(name: str) -> Callable:
    return get_from_config_mapping(
        name,
        {
            'relu': keras.activations.relu,
            'relu6': tf.nn.relu6,
            'leaky_relu': lambda: keras.layers.LeakyReLU(),
            'elu': keras.activations.elu,
            'selu': keras.activations.selu,
            'sigmoid': keras.activations.sigmoid,
            'hard_sigmoid': keras.activations.hard_sigmoid,
            'swish': keras.activations.swish,
            'tanh': keras.activations.tanh,
            'softplus': keras.activations.softplus,
            'softsign': keras.activations.softsign,
            'softmax': keras.activations.softmax,
            'linear': keras.activations.linear,
        },
        'activation',
    )


def get_batch_normalization_factory(name: str) -> Any:
    return get_from_config_mapping(
        name,
        {
            'all': lambda: layers.BatchNormalization(),
            'none': lambda x: x,
        },
        'batch_norm',
    )


def make_conv_network(
    input_shape: List[int],
    downsamples: int,
    widths: List[int],
    input_activation: str,
    internal_activation: str,
    output_activation: str,
    cell_depth: int,
    cell_type: str,
    cell_nodes: int,
    cell_ops: List[List[str]],
    classes: int,
    batch_norm: str,
) -> NetworkModule:
    """ Construct CNN out of NetworkModules. """
    cell_setup = False
    # Determine internal structure
    layer_list = []
    widths_list = []
    cell_depths = [
        cell_depth // (downsamples + 1) for _ in range(downsamples + 1)
    ]
    for i in range(cell_depth % (downsamples + 1)):
        cell_depths[-i - 1] += 1

    for i in range(downsamples + 1):
        # Add downsampling layer
        if i > 0:
            layer_list.append('downsample')
            widths_list.append(widths[i])
        # Add cells
        for _ in range(cell_depths[i]):
            layer_list.append('cell')
            widths_list.append(widths[i])

    input_layer = NInput(shape=input_shape)
    if cell_setup:
        raise NotImplementedError()  # TODO: remove this

    # do the updated keras layer wise construction
    current = make_conv_stem(input_layer, widths[0], batch_norm)
    # Loop through layers
    for i in range(len(layer_list)):
        layer_width = widths_list[i]
        layer_type = layer_list[i]
        if layer_type == 'cell':
            layer = make_cell(
                type=cell_type,
                inputs=current,
                nodes=cell_nodes,
                operations=cell_ops,
                filters=layer_width,
                batch_norm=batch_norm,
                activation=internal_activation,
                # TODO: include regularizers, etc
            )
        elif layer_type == 'downsample':
            layer = make_downsample(
                inputs=current,
                filters=layer_width,
                batch_norm=batch_norm,
                activation=internal_activation,
            )
        else:
            raise ValueError(f'Unknown layer type {layer_type}')
        current = layer
    # Add final classifier
    final_classifier = make_final_classifier(inputs=current,
                                             classes=classes,
                                             activation=output_activation)
    # else:  # Do the outdated cell-wise construction
    #     current = NConvStem(
    #         inputs=[
    #             input_layer,
    #         ],
    #         activation=internal_activation,
    #         filters=widths[0],
    #         batch_norm=batch_norm,
    #         input_channels=input_shape[-1],
    #     )
    #     # Loop through layers
    #     for i in range(len(layer_list)):
    #         layer_width = widths_list[i]
    #         layer_type = layer_list[i]
    #         if layer_type == 'cell':
    #             layer = NCell(
    #                 inputs=[
    #                     current,
    #                 ],
    #                 activation=internal_activation,
    #                 filters=layer_width,
    #                 batch_norm=batch_norm,
    #                 cell_type=cell_type,
    #                 nodes=cell_nodes,
    #                 operations=cell_ops,
    #             )
    #         elif layer_type == 'downsample':
    #             layer = NDownsample(
    #                 inputs=[
    #                     current,
    #                 ],
    #                 activation=internal_activation,
    #                 filters=layer_width,
    #             )
    #         else:
    #             raise ValueError(f'Unknown layer type {layer_type}')
    #         current = layer

    #     # Add final classifier
    #     final_classifier = NFinalClassifier(
    #         inputs=[
    #             current,
    #         ],
    #         activation=output_activation,
    #         classes=classes,
    #     )
    return final_classifier


def compute_network_configuration(num_outputs,
                                  ml_task: str) -> Tuple[Any, Any]:
    output_activation = 'relu'
    if ml_task == 'regression':
        run_loss = losses.mean_squared_error
        output_activation = 'sigmoid'
    elif ml_task == 'classification':
        if num_outputs == 1:
            output_activation = 'sigmoid'
            run_loss = losses.binary_crossentropy
        else:
            output_activation = 'softmax'
            run_loss = losses.categorical_crossentropy
    else:
        raise Exception('Unknown task "{}"'.format(ml_task))

    return output_activation, run_loss


def binary_search_int(
    objective: Callable[[int], Union[int, float]],
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


def find_best_layout_for_budget_and_depth(
        input_shape: Tuple[int, ...], residual_mode: Optional[str],
        input_activation: str, internal_activation: str,
        output_activation: str, target_size: int,
        make_widths: Callable[[int], List[int]],
        layer_args: dict) -> Tuple[int, List[int], NetworkModule]:
    best = (math.inf, None, None)

    def search_objective(w0):
        nonlocal best
        widths = make_widths(w0)
        network = make_network_module_graph(
            input_shape,
            widths,
            residual_mode,
            input_activation,
            internal_activation,
            output_activation,
            layer_args,
        )
        delta = network.num_free_parameters_in_graph - target_size

        if abs(delta) < abs(best[0]):
            best = (delta, widths, network)

        return delta

    binary_search_int(search_objective, 1, int(2**30))
    return best  # type: ignore


def get_rectangular_widths(num_outputs: int,
                           depth: int) -> Callable[[float], List[int]]:

    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.extend((int(round(w0)) for k in range(0, depth - 1)))
        layout.append(num_outputs)
        return layout

    return make_layout


def get_trapezoidal_widths(num_outputs: int,
                           depth: int) -> Callable[[float], List[int]]:

    def make_layout(w0):
        beta = (w0 - num_outputs) / (depth - 1)
        return [int(round(w0 - beta * k)) for k in range(0, depth)]

    return make_layout


def get_exponential_widths(num_outputs: int,
                           depth: int) -> Callable[[float], List[int]]:

    def make_layout(w0):
        beta = math.exp(math.log(num_outputs / w0) / (depth - 1))
        return [
            max(num_outputs, int(round(w0 * (beta**k))))
            for k in range(0, depth)
        ]

    return make_layout


def get_wide_first_layer_rectangular_other_layers_widths(
    num_outputs: int,
    depth: int,
    first_layer_width_multiplier: float = 10,
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


def get_wide_first_2x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 2)


def get_wide_first_4x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 4)


def get_wide_first_8x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 8)


def get_wide_first_16x(num_outputs: int,
                       depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 16)


def get_wide_first_5x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 5)


def get_wide_first_20x(num_outputs: int,
                       depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 20)


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
    elif shape == 'wide_first_8x':
        return get_wide_first_8x
    elif shape in {'wide_first', 'wide_first_10x'}:
        return get_wide_first_layer_rectangular_other_layers_widths
    elif shape == 'wide_first_16x':
        return get_wide_first_16x
    elif shape == 'wide_first_20x':
        return get_wide_first_20x
    else:
        assert False, 'Shape "{}" not recognized.'.format(shape)
