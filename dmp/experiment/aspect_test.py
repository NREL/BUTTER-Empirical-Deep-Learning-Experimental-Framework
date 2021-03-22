"""

"""
import gc
import json
import math
import os
import sys
from copy import deepcopy
from functools import singledispatchmethod
from typing import Callable, Union, List

import numpy
import pandas
import tensorflow
from tensorflow.keras import (
    callbacks,
    metrics,
    optimizers,
)
from tensorflow.python.keras import losses, Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split

from command_line_tools import (
    command_line_config,
    run_tools,
)
from dmp.data.pmlb import pmlb_loader
from dmp.data.pmlb.pmlb_loader import load_dataset
from dmp.experiment.structure.n_add import NAdd
from dmp.experiment.structure.n_fully_connected_layer import NFullyConnectedLayer
from dmp.experiment.structure.n_input import NInput
from dmp.experiment.structure.network_module import NetworkModule
from dmp.data.logging import write_log



def count_trainable_parameters_in_keras_model(model: Model) -> int:
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
        def _(self, target: NFullyConnectedLayer) -> int:
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
) -> NetworkModule:
    # print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))

    input_layer = NInput((), inputs.shape[1:])
    current = input_layer
    #layers = []
    # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
    for d in range(depth):
        layer_width = widths[d]

        activation = internal_activation
        if d == 0:
            activation = input_activation
        elif d == depth - 1:
            activation = output_activation

        layer = NFullyConnectedLayer((current,), (layer_width,), activation)

        if residual_mode == 'none':
            pass
        elif residual_mode == 'full':
            if d > 0 and d < depth-1: # output layer could be of different dimension
                layer = NAdd((layer, current)) ## TODO JP: Only works for rectangle
        else:
            raise Exception('Unknown residual mode "{}".'.format(residual_mode))

        print('d {} w {} in {}'.format(d, layer_width, num_inputs))
        #layers.append(layer)
        current = layer

    return current


def make_model_from_network(target: NetworkModule, model_cache:dict = {}) -> ([any], any):
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
        def _(self, target: NFullyConnectedLayer, inputs: []) -> any:
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
            model_cache[i] = make_model_from_network(i, model_cache)
        i_network_inputs, i_keras_module = model_cache[i]
        for new_input in i_network_inputs:
            network_inputs[new_input.ref()] = new_input
        keras_inputs.append(i_keras_module)

    keras_module, is_input = MakeModelFromNetwork().visit(target, keras_inputs)

    if is_input:
        network_inputs[keras_module.ref()] = keras_module
    
    return list(network_inputs.values()), keras_module




def compute_network_configuration(dataset) -> (any, any):
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

    return output_activation, run_loss


# def make_model_from_network_old(
#         dataset,
#         inputs: numpy.ndarray,
#         outputs: numpy.ndarray,
#         network: NetworkModule,
#         residual_mode: str,
# ) -> Model:
#     num_inputs = inputs.shape[1]
#     num_outputs = outputs.shape[1]
#
#     run_optimizer = optimizers.Adam(0.001)
#     run_metrics = [
#         # metrics.CategoricalAccuracy(),
#         'accuracy',
#         metrics.CosineSimilarity(),
#         metrics.Hinge(),
#         metrics.KLDivergence(),
#         metrics.MeanAbsoluteError(),
#         metrics.MeanSquaredError(),
#         metrics.MeanSquaredLogarithmicError(),
#         metrics.RootMeanSquaredError(),
#         metrics.SquaredHinge(),
#     ]
#
#     print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))
#     # print(inputs[0, :])
#     # print(outputs[0, :])
#
#     run_loss = losses.mean_squared_error
#     output_activation = tensorflow.nn.relu
#     run_task = dataset['Task']
#     if run_task == 'regression':
#         run_loss = losses.mean_squared_error
#         output_activation = tensorflow.nn.sigmoid
#         print('mean_squared_error')
#     elif run_task == 'classification':
#         output_activation = tensorflow.nn.softmax
#         if num_outputs == 1:
#             run_loss = losses.binary_crossentropy
#             print('binary_crossentropy')
#         else:
#             run_loss = losses.categorical_crossentropy
#             print('categorical_crossentropy')
#     else:
#         raise Exception('Unknown task "{}"'.format(run_task))
#
#     input_layer = Input(shape=(num_inputs,))
#     previous = input_layer
#     layers = []
#     for d in range(depth):
#         layer_width = widths[d]
#
#         if d == depth - 1:
#             # output layer
#             activation = output_activation
#         else:
#             activation = tensorflow.nn.relu
#
#         layer = None
#         if d == 0:
#             # input layer
#             layer = tensorflow.keras.layers.Dense(
#                 layer_width,
#                 activation=activation,
#                 # input_shape=(num_inputs,)
#             )(previous)
#
#         else:
#             if residual_mode == 'none':
#                 layer = tensorflow.keras.layers.Dense(
#                     layer_width,
#                     activation=activation,
#                 )()
#             elif residual_mode == 'full':
#                 layer = tensorflow.keras.layers.Dense(
#                     layer_width,
#                     activation=activation,
#                 )(previous)
#                 layer = Add()(previous, layer)
#             else:
#                 raise Exception('Unknown residual mode "{}".'.format(residual_mode))
#
#         print('d {} w {} in {}'.format(d, layer_width, num_inputs))
#         layers.append(layer)
#         previous = layer
#
#     # model = Sequential(layers)
#     model = Model(inputs=input_layer, outputs=layers[-1])
#     # model.compile(
#     #     # loss='binary_crossentropy', # binary classification
#     #     # loss='categorical_crossentropy', # categorical classification (one hot)
#     #     loss=run_loss,  # regression
#     #     optimizer=run_optimizer,
#     #     # optimizer='rmsprop',
#     #     # metrics=['accuracy'],
#     #     metrics=run_metrics,
#     # )
#
#     # count_trainable_parameters(model)
#     return model


def test_network(
        config: {},
        dataset,
        inputs: numpy.ndarray,
        outputs: numpy.ndarray,
        keras_input,
        keras_output,
        network:NetworkModule,
) -> dict:
    """
    test_network

    Given a fully constructed Keras network, train and test it on a given dataset using hyperparameters in config
    This function also creates log events during and after training.
    """
    config = deepcopy(config)

    # wine_quality_white__wide_first__4194304__4__16106579275625
    name = '{}__{}__{}__{}'.format(
        dataset['Dataset'],
        config['topology'],
        config['budget'],
        config['depth'],
    )

    config['name'] = name


    log_data = {'config': config}
    run_config = config['run_config']

    run_name = run_tools.get_run_name(config)
    config['run_name'] = run_name

    depth = len(widths)
    config['depth'] = depth
    config['num_hidden'] = max(0, depth - 2)
    config['widths'] = widths


    log_data = {'config': config}
    run_config = config['run_config']

    run_optimizer = optimizers.get(config['optimizer'])
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
    model = Model(inputs=keras_input, outputs=keras_output)

    ## TODO: Would holding off on this step obviate the need for NetworkModule?
    model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss=run_loss,  # regression
        optimizer=run_optimizer,
        # optimizer='rmsprop',
        # metrics=['accuracy'],
        metrics=run_metrics,
    )

    gc.collect()

    assert count_num_free_parameters(network) == count_trainable_parameters_in_keras_model(model), "Wrong number of trainable parameters"

    log_data['num_weights'] = count_trainable_parameters_in_keras_model(model)
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

    print(keras_input)
    print(inputs.shape)

    #### TRAIN THE MODEL
    if config["test_split"] > 0:
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=config["test_split"])
    else:
        inputs_train, outputs_train = inputs, outputs

    history_callback = model.fit(
        x=inputs_train,
        y=outputs_train,
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

    #### TEST THE MODEL TODO JP: Implement evaluation
    if config["test_split"] > 0:
        #evaluate = model.evaluate(x=inputs_test, y=outputs_test)
        #log_data["test_loss"]...
        pass

    log_data['run_name'] = run_name

    return log_data
    


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
        inputs,
        residual_mode,
        input_activation,
        internal_activation,
        output_activation,
        budget,
        make_widths: Callable[[int], List[int]],
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
                               )
        delta = count_num_free_parameters(network) - budget

        if abs(delta) < abs(best[0]):
            best = (delta, widths, network)

        return delta

    binary_search_int(search_objective, 1, int(2 ** 30))
    return best


def get_rectangular_widths(num_outputs, depth) -> Callable[[float], List[int]]:
    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.extend((round(w0) for k in range(0, depth - 1)))
        layout.append(num_outputs)
        return layout

    return make_layout


def get_trapezoidal_widths(num_outputs, depth) -> Callable[[float], List[int]]:
    def make_layout(w0):
        beta = (w0 - num_outputs) / (depth - 1)
        return [round(w0 - beta * k) for k in range(0, depth)]

    return make_layout


def get_exponential_widths(num_outputs, depth) -> Callable[[float], List[int]]:
    def make_layout(w0):
        beta = math.exp(math.log(num_outputs / w0) / (depth - 1))
        return [round(w0 * (beta ** k)) for k in range(0, depth)]

    return make_layout


def get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, first_layer_width_multiplier=10,
) -> Callable[[float], List[int]]:
    def make_layout(w0):
        layout = []
        if depth > 1:
            layout.append(w0)
        if depth > 2:
            inner_width = max(1, round(w0 / first_layer_width_multiplier))
            layout.extend((inner_width for k in range(0, depth - 2)))
        layout.append(num_outputs)
        return layout

    return make_layout


pandas.set_option("display.max_rows", None, "display.max_columns", None)
datasets = pmlb_loader.load_dataset_index()

# core_config = tensorflow.Conf()
# core_config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=core_config)
# tensorflow.keras.backend.set_session(session)

# targets:
# datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white',]
# 'topologies': ['rectangle', 'trapezoid', 'exponential', 'wide_first'],
# 'budgets': [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
#                 8388608, 16777216, 33554432],
# 'depths': [2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20],


default_config = {
    'seed': 42,
    'log': './log',
    'dataset': 'wine_quality_white',
    'test_split': 0,
    'activation': 'relu',
    'optimizer': {
        "class_name": "adam",
        "config": {"learning_rate": 0.001},
    },
    'topologies': ['exponential'],
    'budgets': [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                8388608, 16777216, 33554432],
    'depths': [2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    'residual_modes': ['none', ],
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
        'validation_split': .2, # This is relative to the training set size.
        'shuffle': True,
        'epochs': 10000,
        'batch_size': 256,
    },
}

# example command line use with slurm:
# sbatch -n1 -t8:00:00 --gres=gpu:1 ./srundmp.sh dmp.experiment.aspect_test.py "{'dataset' : mnist, 'budgets' : [ 32768 ], 'topologies' : [ trapezoid ], 'depths' : [ 4 ], 'reps' : 19 }"
# sbatch -n1 -t18:00:00 --gres=gpu:1 ./srundmp.sh dmp.experiment.aspect_test.py "{'dataset': 'mnist','budgets':[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'topologies' : [ 'rectangle' ]}
# results on eagle : /projects/dmpapps/ctripp/data/log

# example command line use on laptop:
# conda activate dmp
# python -u -m dmp.experiment.aspect_test "{'dataset' : mnist, 'budgets' : [ 32768 ], 'topologies' : [ trapezoid ], 'depths' : [ 4 ], 'reps' : 19 }"

def initialize_global_state(config):
    global dataset, inputs, outputs
    dataset, inputs, outputs = load_dataset(datasets, config['dataset'])
    gc.collect()
    global num_observations, num_inputs, num_outputs
    num_observations = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

def aspect_test(config:dict) -> dict:
    """
    Programmatic interface to the main functionality of this module. Run an aspect test for a single configuration and return the result as a dictionary.
    """

    numpy.random.seed(config["seed"])
    tensorflow.random.set_seed(config["seed"])

    if "dataset" not in globals():
        initialize_global_state(config)

    global topology, residual_mode, budget, depth
    global output_activation, run_loss, delta, widths, network

    topology = config['topology']
    residual_mode = config['residual_mode']
    budget = config['budget']
    depth = config['depth']

    widths_factory = None
    if topology == 'rectangle':
        widths_factory = get_rectangular_widths
    elif topology == 'trapezoid':
        widths_factory = get_trapezoidal_widths
    elif topology == 'exponential':
        widths_factory = get_exponential_widths
    elif topology == 'wide_first':
        widths_factory = get_wide_first_layer_rectangular_other_layers_widths
    else:
        assert False, 'Topology "{}" not recognized.'.format(topology)

    output_activation, run_loss = compute_network_configuration(dataset)
    delta, widths, network = find_best_layout_for_budget_and_depth(
        inputs,
        residual_mode,
        tensorflow.nn.relu,
        tensorflow.nn.relu,
        output_activation,
        budget,
        widths_factory(num_outputs, depth))
    config['widths'] = widths
    reps = config['reps']

    print('begin reps: budget: {}, depth: {}, widths: {}, reps: {}'.format(budget, depth, widths, reps))

    
    keras_inputs, keras_output = make_model_from_network(network)

    assert len(keras_inputs) == 1, 'Wrong number of keras inputs generated'
    keras_input = keras_inputs[0]

    return test_network(config, dataset, inputs, outputs, keras_input, keras_output, network)

def run_multiple_aspect_tests_from_config(config):
    """
    Entrypoint for the primary use of this module. Take a config that describes many different potential runs, and loop through them in series - logging data after each run. 
    """
    initialize_global_state(config)

    for topology in config['topologies']:
        config['topology'] = topology
        for residual_mode in config['residual_modes']:
            config['residual_mode'] = residual_mode
            for budget in config['budgets']:
                config['budget'] = budget
                for depth in config['depths']:
                    config['depth'] = depth
                    
                    log_data = aspect_test(config)
                    write_log(log_data, config['log'])
                    gc.collect()

                    # for rep in range(reps):
                    #     print('begin rep {}...'.format(rep))
                    #     this_config = deepcopy(config)
                    #     this_config['datasetName'] = dataset['Dataset']
                    #     this_config['datasetRow'] = list(dataset)
                    #
                    #     model = make_network_model(dataset, inputs, outputs, widths, residual_mode)
                    #
                    #     test_network(config, dataset, inputs, outputs, model)
                    #     gc.collect()
    print('done.')

## Support the python -m runpy interface
if __name__ == "__main__":
    config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
    run_multiple_aspect_tests_from_config(config)
