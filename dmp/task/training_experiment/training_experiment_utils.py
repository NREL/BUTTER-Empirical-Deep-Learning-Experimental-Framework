import math
from multiprocessing import pool
import random
import time
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy
import tensorflow.keras as keras
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
from dmp.layer.conv_cell import make_graph_cell, make_parallel_add_cell

from dmp.layer.visitor.compute_free_parameters import compute_free_parameters
from dmp.layer import *


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


def split_dataset(
    test_split_method: str,
    test_split: float,
    validation_split: float,
    label_noise: float,
    run_task: str,
    inputs,
    outputs,
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any], ]:
    validation_inputs, validation_outputs = (None, None)
    train_inputs, train_outputs = (None, None)
    test_inputs, test_outputs = (None, None)

    if test_split_method == 'shuffled_train_test_split':
        train_inputs, test_inputs, train_outputs, test_outputs = \
                train_test_split(
                    inputs,
                    outputs,
                    test_size=test_split,
                    shuffle=True,
                )

        if validation_split is not None and validation_split > 0.0:
            train_inputs, validation_inputs, train_outputs, validation_outputs = \
                train_test_split(
                    train_inputs,
                    train_outputs,
                    test_size=int(validation_split/(1-test_split)),
                    shuffle=True,
                )

        add_label_noise(label_noise, run_task, train_outputs)
    else:
        raise NotImplementedError(
            f'Unknown test_split_method {test_split_method}.')

    return (
        (train_inputs, train_outputs),
        (validation_inputs, validation_outputs),
        (test_inputs, test_outputs),
    )


# def make_simple_dense_network(
#     input_shape: Sequence[int],
#     widths: List[int],
#     residual_mode: Optional[str],
#     # input_activation: str,
#     output_activation: str,
#     layer_config: Dict[str, Any],
# ) -> Layer:
#     # print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))

#     parent = Input({'shape': input_shape}, [])
#     # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
#     for depth, width in enumerate(widths):
#         config = layer_config.copy()

#         # Activation functions may be different for input, output, and hidden layers
#         # if depth == 0:
#         #     config['activation'] = input_activation
#         if depth == len(widths) - 1:
#             config['activation'] = output_activation

#         # Fully connected layer
#         config['units'] = width
#         layer = Dense(config, parent)

#         # Skip connections for residual modes
#         if residual_mode is None or residual_mode == 'none':
#             pass
#         elif residual_mode == 'full':
#             # If this isn't the first or last layer, and the previous layer is
#             # of the same width insert a residual sum between layers
#             # NB: Only works for rectangle
#             if depth > 0 and depth < len(widths) - 1 and width == widths[depth
#                                                                          - 1]:
#                 layer = Add({}, [layer, parent])
#         else:
#             raise NotImplementedError(
#                 f'Unknown residual mode "{residual_mode}".')
#         parent = layer
#     return parent


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
    config: Optional[Dict],
    mapping: Dict[str, Callable],
    config_name: str,
    *args,
    **kwargs,
) -> Any:
    if config is None:
        return None

    type, params = get_params_and_type_from_config(config)
    factory = get_from_config_mapping(type, mapping, config_name)
    return factory(*args, **kwargs, **params)


def get_activation_factory(name: str) -> Callable:
    return get_from_config_mapping(
        name,
        {
            'relu': keras.activations.relu,
            'relu6': tensorflow.nn.relu6,
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
            'all': lambda: keras.layers.BatchNormalization(),
            'none': lambda x: x,
        },
        'batch_norm',
    )


'''
+ Purpose?
    + Make CNN/NN based on simplified task parameters
    + Find NN that fits specific size using a shape function

-> don't need intermediate 'widths here'
    -> make a 'widths' extractor to log widths instead
    -> instead work directly off of shape function?
        -> shapeFunc(search parameter) -> network -> size_visitor -> size
            -> shape func for nn's could make a widths-list, or not
            -> maybe direct construction is best?
            -> function that outputs layer type and width?
            -> **function that outputs layer**
'''


def make_cnn_network(
    input_shape: Sequence[int],
    num_stacks: int,
    cells_per_stack: int,
    stem_factory,
    cell_factory,
    downsample_factory,
    pooling_factory,
    output_factory,
) -> Layer:
    '''
    + Structure:
        + Stem: 3x3 Conv with input activation
        + Repeat N times:
            + Repeat M times:
                + Cell
            + Downsample and double channels, optionally with Residual Connection
                + Residual: 2x2 average pooling layer with stride 2 and a 1x1
        + Global Pooling Layer
        + Dense Output Layer with output activation

    + Total depth (layer-wise or stage-wise)
    + Total number of cells
    + Total number of downsample steps
    + Number of Cells per downsample
    + Width / num channels
        + Width profile (non-rectangular widths)
    + Cell choice
    + Downsample choice (residual mode, pooling type)

    + config code inputs:
        + stem factory?
        + cell factory
        + downsample factory
        + pooling factory
        + output factory?
    '''

    layer: Layer = Input({'shape': input_shape}, [])
    layer = stem_factory(layer)
    for s in range(num_stacks):
        for c in range(cells_per_stack):
            layer = cell_factory(layer)
        layer = downsample_factory(layer)
    layer = pooling_factory(layer)
    layer = output_factory(layer)
    return layer


def stem_generator(
    filters: int,
    conv_config: Dict[str, Any],
) -> Callable[[Layer], Layer]:
    return lambda input: DenseConv.make(filters, (3, 3),
                                        (1, 1), conv_config, input)


def cell_generator(
        width: int,  # width of input and output
        operations: List[List[str]],  # defines the cell structure
        conv_config: Dict[str, Any],  # conv layer configuration
        pooling_config: Dict[str, Any],  # pooling layer configuration
) -> Callable[[Layer], Layer]:
    return lambda input: make_graph_cell(width, operations, conv_config,
                                         pooling_config, input)


def residual_downsample_generator(
    conv_config: Dict[str, Any],
    pooling_config: Dict[str, Any],
    width: int,
    stride: Sequence[int],
) -> Callable[[Layer], Layer]:

    def factory(input: Layer) -> Layer:
        long = DenseConv.make(width, (3, 3), stride, conv_config, input)
        long = DenseConv.make(width, (3, 3), (1, 1), conv_config, long)
        short = APoolingLayer.make(\
            AvgPool, (2, 2), (2, 2), pooling_config, input)
        return Add({}, [long, short])

    return factory


def output_factory_generator(
    config: Dict[str, Any],
    num_outputs: int,
) -> Callable[[Layer], Layer]:
    return lambda input: Dense(config, GlobalAveragePooling({}, input))


def get_output_activation_and_loss_for_ml_task(
    num_outputs,
    ml_task: str,
) -> Tuple[Any, Any]:
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
    target_num_free_parameters: int, make_network: Callable[[int], Layer]
) -> Tuple[int, Layer, int, Dict[Layer, Tuple]]:
    best = (math.inf, None, None)

    def search_objective(search_parameter):
        nonlocal best
        network = make_network(search_parameter)
        num_free_parameters, layer_shapes = compute_free_parameters(network)
        delta = num_free_parameters - target_num_free_parameters

        if abs(delta) < abs(best[0]):
            best = (delta, network, num_free_parameters, layer_shapes)

        return delta

    binary_search_int(search_objective, 1, int(2**31))
    return best  # type: ignore


def get_rectangular_widths(num_outputs: int,
                           depth: int) -> Callable[[float], List[int]]:

    def make_layout(search_parameter):
        layout = []
        if depth > 1:
            layout.extend((int(round(search_parameter)) for k in range(0, depth - 1)))
        layout.append(num_outputs)
        return layout

    return make_layout


def get_trapezoidal_widths(num_outputs: int,
                           depth: int) -> Callable[[float], List[int]]:

    def make_layout(search_parameter):
        beta = (search_parameter - num_outputs) / (depth - 1)
        return [int(round(search_parameter - beta * k)) for k in range(0, depth)]

    return make_layout


def get_exponential_widths(num_outputs: int,
                           depth: int) -> Callable[[float], List[int]]:

    def make_layout(search_parameter):
        beta = math.exp(math.log(num_outputs / search_parameter) / (depth - 1))
        return [
            max(num_outputs, int(round(search_parameter * (beta**k))))
            for k in range(0, depth)
        ]

    return make_layout


def get_wide_first_layer_rectangular_other_layers_widths(
    num_outputs: int,
    depth: int,
    first_layer_width_multiplier: float = 10,
) -> Callable[[float], List[int]]:

    def make_layout(search_parameter):
        layout = []
        if depth > 1:
            layout.append(search_parameter)
        if depth > 2:
            inner_width = max(1, int(round(search_parameter / first_layer_width_multiplier)))
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
