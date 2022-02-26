"""

"""
import gc
import json
import math
import os
import sys
from copy import deepcopy
from typing import Callable, Union, List

import numpy
import pandas
import tensorflow
from tensorflow.keras import (
    callbacks,
    losses,
    Sequential,
    metrics,
    optimizers,
)
from tensorflow.python.keras.models import Model

from command_line_tools import (
    command_line_config,
    run_tools,
)
from dmp.data.pmlb import pmlb_loader
from dmp.data.pmlb.pmlb_loader import load_dataset


def count_trainable_parameters(model: Model) -> int:
    count = 0
    for var in model.trainable_variables:
        print('ctp {}'.format(var.get_shape()))
        acc = 1
        for dim in var.get_shape():
            acc *= int(dim)
        print('ctp acc {}'.format(acc))
        count += acc
    print('ctp total {}'.format(count))
    return count


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def test_network(
        config: {},
        dataset,
        inputs: numpy.ndarray,
        outputs: numpy.ndarray,
        widths: [int],
) -> None:
    config = deepcopy(config)
    depth = len(widths)

    config['depth'] = depth
    config['num_hidden'] = max(0, depth - 2)
    config['widths'] = widths

    #wine_quality_white__wide_first__4194304__4__16106579275625
    name = '{}__{}__{}__{}'.format(
        dataset['Dataset'],
        config['topology'],
        config['budget'],
        config['depth'],
    )

    config['name'] = name

    # pprint(config)
    run_name = run_tools.get_run_name(config)
    config['run_name'] = run_name

    num_observations = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    log_data = {'config': config}
    run_config = config['run_config']

    run_optimizer = optimizers.Adam(0.001)
    run_metrics = [
        # metrics.CategoricalAccuracy(),
        'accuracy',
        metrics.CosineSimilarity(),
        metrics.Hinge(),
        metrics.KLDivergence(),
        metrics.MeanAbsoluteError(),
        metrics.MeanSquaredError(),
        metrics.MeanSquaredLogarithmicError(),
        metrics.RootMeanSquaredError(),
        metrics.SquaredHinge(),
    ]

    print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))
    # print(inputs[0, :])
    # print(outputs[0, :])
    run_loss = losses.mean_squared_error
    output_activation = tensorflow.nn.relu
    run_task = dataset['Task']
    if run_task == 'regression':
        run_loss = losses.mean_squared_error
        output_activation = tensorflow.nn.sigmoid
        print('mean_squared_error')
    elif run_task == 'classification':
        output_activation = tensorflow.nn.softmax
        if num_outputs == 1:
            run_loss = losses.binary_crossentropy
            print('binary_crossentropy')
        else:
            run_loss = losses.categorical_crossentropy
            print('categorical_crossentropy')
    else:
        raise Exception('Unknown task "{}"'.format(run_task))

    layers = []
    for d in range(depth):
        layer_width = widths[d]

        if d == depth - 1:
            # output layer
            activation = output_activation
        else:
            activation = tensorflow.nn.relu

        layer = None
        if d == 0:
            # input layer
            layer = tensorflow.keras.layers.Dense(
                layer_width,
                activation=activation,
                input_shape=(num_inputs,))
        else:
            layer = tensorflow.keras.layers.Dense(
                layer_width,
                activation=activation,
            )
        print('d {} w {} in {}'.format(d, layer_width, num_inputs))
        layers.append(layer)

    model = Sequential(layers)
    model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss=run_loss,  # regression
        optimizer=run_optimizer,
        # optimizer='rmsprop',
        # metrics=['accuracy'],
        metrics=run_metrics,
    )

    log_data['num_weights'] = count_trainable_parameters(model)
    log_data['num_inputs'] = num_inputs
    log_data['num_features'] = dataset['n_features']
    log_data['num_classes'] = dataset['n_classes']
    log_data['num_outputs'] = num_outputs
    log_data['num_observations'] = num_observations
    log_data['task'] = dataset['Task']
    log_data['endpoint'] = dataset['Endpoint']

    run_callbacks = [
        callbacks.EarlyStopping(**config['early_stopping']),
    ]

    gc.collect()

    history_callback = model.fit(
        x=inputs,
        y=outputs,
        callbacks=run_callbacks,
        **run_config,
    )

    history = history_callback.history
    log_data['history'] = history

    validation_losses = numpy.array(history['val_loss'])
    best_index = numpy.argmin(validation_losses)

    log_data['iterations'] = best_index + 1
    log_data['val_loss'] = validation_losses[best_index]
    log_data['loss'] = history['loss'][best_index]

    log_path = config['log']
    run_tools.makedir_if_not_exists(log_path)
    log_file = os.path.join(log_path, '{}.json'.format(run_name))
    print('log file: {}'.format(log_file))

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, sort_keys=True, cls=NpEncoder)


# def get_rectangular_layout(inputs, outputs, budget, depth) -> [int]:
#     i = inputs.shape[1]
#     h = (depth - 2)
#     o = outputs.shape[1]
#     a = h
#     b = i + h + o + 1
#     c = o - budget
#     raw_width = 1
#     if h == 0:
#         raw_width = -(o - budget) / (i + o + 1)
#     else:
#         raw_width = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
#     width = round(raw_width)
#     print(
#         'budget {} depth {}, i {} h {} o {}, a {} b {} c {}, raw_width {}, width {}'.format(
#             budget, depth, i, h, o, a, b, c, raw_width, width))
#
#     widths = []
#     for d in range(depth):
#         if d == depth - 1:  # output layer
#             layerWidth = num_outputs
#         else:
#             layerWidth = width
#         widths.append(layerWidth)
#     return widths


# def binary_search_float(objective: Callable[[float], any],
#                         minimum: float,
#                         maximum: float,
#                         max_iters: int = 32,
#                         threshold: float = 1e-3,
#                         ) -> (float, bool):
#     """
#     :param objective: function for which to find fixed point
#     :param minimum: min value of search
#     :param maximum: max value of search
#     :param max_iters: max iterations
#     :param threshold: distance between max and min search points upon which to exit early
#     :return: solution
#     """
#     if minimum > maximum:
#         raise ValueError("binary search minimum must be less than maximum")
#     candidate = 0.0
#     for i in range(max_iters):
#         candidate = (maximum + minimum) / 2
#         evaluation = objective(candidate)
#
#         if fabs(maximum - minimum) < threshold:
#             return candidate, True
#
#         if evaluation < 0:  # candidate < target
#             minimum = candidate
#         elif evaluation > 0:  # candidate > target
#             maximum = candidate
#
#     return candidate, False


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


def count_fully_connected_parameters(widths: [int]) -> int:
    depth = len(widths)
    p = (num_inputs + 1) * widths[0]
    if depth > 1:
        for k in range(1, depth):
            p += (widths[k - 1] + 1) * widths[k]
    return p


def find_best_layout_for_budget_and_depth(
        num_inputs,
        num_outputs,
        budget,
        depth,
        make_layout: Callable[[int], List[int]],
) -> [int]:
    def search_objective(w0):
        return count_fully_connected_parameters(make_layout(w0)) - budget

    w0, found = binary_search_int(search_objective, 1, int(2 ** 30))
    widths = make_layout(w0)
    return widths


def get_rectangular_layout(num_inputs, num_outputs, budget, depth) -> [int]:
    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.extend((round(w0) for k in range(0, depth - 1)))
        layout.append(num_outputs)
        return layout

    return find_best_layout_for_budget_and_depth(num_inputs, num_outputs, budget, depth, make_layout)


def get_trapezoidal_layout(num_inputs, num_outputs, budget, depth) -> [int]:
    def make_layout(w0):
        beta = (w0 - num_outputs) / (depth - 1)
        return [round(w0 - beta * k) for k in range(0, depth)]

    return find_best_layout_for_budget_and_depth(num_inputs, num_outputs, budget, depth, make_layout)


def get_exponential_layout(num_inputs, num_outputs, budget, depth) -> [int]:
    def make_layout(w0):
        beta = math.exp(math.log(num_outputs / w0) / (depth - 1))
        return [round(w0 * (beta ** k)) for k in range(0, depth)]

    return find_best_layout_for_budget_and_depth(num_inputs, num_outputs, budget, depth, make_layout)


def get_wide_first_layer_rectangular_other_layers_layout(
        num_inputs,
        num_outputs,
        budget,
        depth,
        first_layer_width_multiplier=10,
) -> [int]:
    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.append(w0)
        if depth > 2:
            inner_width = max(1, round(w0 / first_layer_width_multiplier))
            layout.extend((inner_width for k in range(0, depth - 2)))
        layout.append(num_outputs)
        return layout

    return find_best_layout_for_budget_and_depth(num_inputs, num_outputs, budget, depth, make_layout)


pandas.set_option("display.max_rows", None, "display.max_columns", None)
datasets = pmlb_loader.load_dataset_index()

# core_config = tensorflow.Conf()
# core_config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=core_config)
# tensorflow.keras.backend.set_session(session)

default_config = {
    'log': './log',
    'dataset': 'wine_quality_white',
    'activation': 'relu',
    'topologies': ['wide_first'],
    'budgets': [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608],
    'depths': [2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    'residual_modes': ['none', 'full'],
    'reps': 30,
    'early_stopping': {
        'patience': 10,
        'monitor': 'val_loss',
        'min_delta': 0,
        'verbose': 0,
        'mode': 'min',
        'baseline': None,
        'restore_best_weights': False,
    },
    'run_config': {
        'validation_split': .2,
        'shuffle': True,
        'epochs': 10000,
        'batch_size': 256,
    },
}

# example command line use with slurm:
# sbatch -n1 -t8:00:00 --gres=gpu:1 ./srundmp.sh dmp.aspect_test.py "{'dataset' : mnist, 'budgets' : [ 32768 ], 'topologies' : [ trapezoid ], 'depths' : [ 4 ], 'reps' : 19 }"
# sbatch -n1 -t18:00:00 --gres=gpu:1 ./srundmp.sh dmp.aspect_test.py "{'dataset': 'mnist','budgets':[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'topologies' : [ 'rectangle' ]}
# results on eagle : /projects/dmpapps/ctripp/data/log

# example command line use on laptop:
# conda activate dmp
# python -u -m dmp.aspect_test "{'dataset' : mnist, 'budgets' : [ 32768 ], 'topologies' : [ trapezoid ], 'depths' : [ 4 ], 'reps' : 19 }"

config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)

dataset, inputs, outputs = load_dataset(datasets, config['dataset'])
gc.collect()

num_observations = inputs.shape[0]
num_inputs = inputs.shape[1]
num_outputs = outputs.shape[1]

for topology in config['topologies']:
    config['topology'] = topology
    for residual_mode in config['residual_modes']:
        config['residual_mode'] = residual_mode
        for budget in config['budgets']:
            config['budget'] = budget
            for depth in config['depths']:
                config['depth'] = depth

                widths = []
                if topology == 'rectangle':
                    widths = get_rectangular_layout(num_inputs, num_outputs, budget, depth)
                elif topology == 'trapezoid':
                    widths = get_trapezoidal_layout(num_inputs, num_outputs, budget, depth)
                elif topology == 'exponential':
                    widths = get_exponential_layout(num_inputs, num_outputs, budget, depth)
                elif topology == 'wide_first':
                    widths = get_wide_first_layer_rectangular_other_layers_layout(num_inputs, num_outputs, budget, depth)
                else:
                    assert False, 'Topology "{}" not recognized.'.format(topology)
                config['widths'] = widths

                reps = config['reps']

                print('begin reps: budget: {}, depth: {}, widths: {}, reps: {}'.format(budget, depth, widths, reps))

                for _ in range(reps):
                    this_config = deepcopy(config)
                    this_config['datasetName'] = dataset['Dataset']
                    this_config['datasetRow'] = list(dataset)

                    test_network(config, dataset, inputs, outputs, widths)
                    gc.collect()

print('done.')
gc.collect()
