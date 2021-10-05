"""

"""
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


def make_model_from_network(target: NetworkModule, model_cache: dict = {}) -> ([any], any):
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
            model_cache[i] = make_model_from_network(i, model_cache)
        i_network_inputs, i_keras_module = model_cache[i]
        for new_input in i_network_inputs:
            network_inputs[new_input.ref()] = new_input
        keras_inputs.append(i_keras_module)

    keras_module, is_input = MakeModelFromNetwork().visit(target, keras_inputs)

    if is_input:
        network_inputs[keras_module.ref()] = keras_module

    return list(network_inputs.values()), keras_module


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


def test_network(
        config: {},
        dataset,
        inputs: numpy.ndarray,
        outputs: numpy.ndarray,
        keras_input,
        keras_output,
        network: NetworkModule,
        run_loss,
        widths
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

    run_name = run_tools.get_run_name(config)
    config['run_name'] = run_name

    depth = len(widths)
    config['depth'] = depth
    config['num_hidden'] = max(0, depth - 2)
    config['widths'] = widths

    log_data = {'config': config}

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

    print(f'{count_num_free_parameters(network)} {count_trainable_parameters_in_keras_model(model)}')
    assert count_num_free_parameters(network) == count_trainable_parameters_in_keras_model(model), \
        "Wrong number of trainable parameters"

    log_data['num_weights'] = count_trainable_parameters_in_keras_model(model)
    log_data['num_inputs'] = inputs.shape[1]
    log_data['num_features'] = dataset['n_features']
    log_data['num_classes'] = dataset['n_classes']
    log_data['num_outputs'] = outputs.shape[1]
    log_data['num_observations'] = inputs.shape[0]
    log_data['task'] = dataset['Task']
    log_data['endpoint'] = dataset['Endpoint']

    run_callbacks = []
    if config['early_stopping'] != False:
        run_callbacks.append(callbacks.EarlyStopping(**config['early_stopping']))

    run_config = config['run_config'].copy()

    if config["test_split"] > 0:
        ## train/test/val split
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs,
                                                                                  test_size=config["test_split"])
        run_config["validation_split"] = run_config["validation_split"] / (1 - config["test_split"])

        ## Set up a custom callback to record test loss at each epoch
        ## This could potentially cause performance issues with large datasets on GPU
        class TestHistory(Callback):
            def __init__(self, x_test, y_test):
                self.x_test = x_test
                self.y_test = y_test

            def on_train_begin(self, logs={}):
                self.history = {}

            def on_epoch_end(self, epoch, logs=None):
                eval_log = self.model.evaluate(x=self.x_test, y=self.y_test, return_dict=True)
                for k, v in eval_log.items():
                    k = "test_" + k
                    self.history.setdefault(k, []).append(v)

        test_history_callback = TestHistory(inputs_test, outputs_test)
        run_callbacks.append(test_history_callback)
    else:
        ## Just train/val split
        inputs_train, outputs_train = inputs, outputs

    if "tensorboard" in config.keys():
        run_callbacks.append(TensorBoard(
            log_dir=os.path.join(config["tensorboard"], run_name),
            # append ", config["residual_mode"]" to add resisual to tensorboard path
            histogram_freq=1
        ))

    if "plot_model" in config.keys():
        if not os.path.exists(config["plot_model"]):
            os.makedirs(config["plot_model"])
        tensorflow.keras.utils.plot_model(
            model,
            to_file=os.path.join(config["plot_model"], run_name + ".png"),
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    # TRAINING
    # run_config["verbose"] = 0  # This overrides verbose logging.

    ## Checkpoint Code
    if "checkpoint_epochs" in config.keys():

        assert config["test_split"] == 0, "Checkpointing is not compatible with test_split."

        DMP_CHECKPOINT_DIR = os.getenv("DMP_CHECKPOINT_DIR", default="checkpoints")
        if "checkpoint_dir" in config.keys():
            DMP_CHECKPOINT_DIR = config["checkpoint_dir"]
        if not os.path.exists(DMP_CHECKPOINT_DIR):
            os.makedirs(DMP_CHECKPOINT_DIR)

        if "jq_uuid" in config.keys():
            checkpoint_name = config["jq_uuid"]
        else:
            checkpoint_name = run_name

        model = ResumableModel(model,
                               save_every_epochs=config["checkpoint_epochs"],
                               to_path=os.path.join(DMP_CHECKPOINT_DIR, checkpoint_name + ".h5"))

    history = model.fit(
        x=inputs_train,
        y=outputs_train,
        callbacks=run_callbacks,
        **run_config,
    )

    if not "checkpoint_epochs" in config.keys():
        # Tensorflow models return a History object from their fit function, but ResumableModel objects returns History.history. This smooths out that incompatibility.
        history = history.history

    # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
    # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
    # model.save_weights(f"./log/weights/{run_name}.h5", save_format="h5")
    # model.save(f"./log/models/{run_name}.h5", save_format="h5")

    log_data['history'] = history

    validation_losses = numpy.array(history['val_loss'])
    best_index = numpy.argmin(validation_losses)

    log_data['iterations'] = best_index + 1
    log_data['val_loss'] = validation_losses[best_index]
    log_data['loss'] = history['loss'][best_index]

    if config["test_split"] > 0:
        ## Record the history of test evals from the callback
        log_data['history'].update(test_history_callback.history)
        test_losses = numpy.array(history['test_loss'])
        log_data['test_loss'] = test_losses[best_index]

    log_data['run_name'] = run_name

    log_data['environment'] = {
        "tensorflow_version": tensorflow.__version__,
    }

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
    'mode': 'list',  # 'direct', 'list', 'enqueue', ?
    'seed': None,
    'log': './log',
    'dataset': '529_pollen',
    'test_split': 0,
    'activation': 'relu',
    'optimizer': {
        "class_name": "adam",
        "config": {"learning_rate": 0.001},
    },
    'datasets': ['529_pollen'],
    'learning_rates': [0.001],
    'topologies': ['wide_first'],
    'budgets': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                8388608, 16777216, 33554432],
    'depths': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    'epoch_scale': {
        'm': 0,
        'b': numpy.log(3001),
    },
    'residual_modes': ['none', ],
    'reps': 10,
    'early_stopping': False,
    'run_config': {
        'validation_split': .2,  # This is relative to the training set size.
        'shuffle': True,
        'epochs': 3000,
        'batch_size': 256,
        'verbose': 0,
    },
}


# example command line use with slurm:
# sbatch -n1 -t8:00:00 --gres=gpu:1 ./srundmp.sh dmp.experiment.aspect_test.py "{'dataset' : mnist, 'budgets' : [ 32768 ], 'topologies' : [ trapezoid ], 'depths' : [ 4 ], 'reps' : 19 }"
# sbatch -n1 -t18:00:00 --gres=gpu:1 ./srundmp.sh dmp.experiment.aspect_test.py "{'dataset': 'mnist','budgets':[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'topologies' : [ 'rectangle' ]}
# results on eagle : /projects/dmpapps/ctripp/data/log

# example command line use on laptop:
# conda activate dmp
# python -u -m dmp.experiment.aspect_test "{'dataset' : mnist, 'budgets' : [ 32768 ], 'topologies' : [ trapezoid ], 'depths' : [ 4 ], 'reps' : 19 }"

def aspect_test(config: dict, strategy: Optional[tensorflow.distribute.Strategy] = None) -> dict:
    """
    Programmatic interface to the main functionality of this module. Run an aspect test for a single configuration and return the result as a dictionary.
    """

    if strategy is None:
        strategy = tensorflow.distribute.get_strategy()

    numpy.random.seed(config["seed"])
    tensorflow.random.set_seed(config["seed"])
    random.seed(config["seed"])

    ## Load dataset
    dataset, inputs, outputs = load_dataset(datasets, config['dataset'])

    # print(f'outputs.shape {outputs.shape} inputs.shape {inputs.shape} task {dataset["Task"]}')
    # print(f'inputs {inputs}')
    # print(f'outputs {outputs}')

    num_outputs = outputs.shape[1]
    topology = config['topology']
    depth = config['depth']
    budget = config['budget']

    ## Network configuration
    output_activation, run_loss = compute_network_configuration(num_outputs, dataset)

    ## Build NetworkModule network
    delta, widths, network = find_best_layout_for_budget_and_depth(
        inputs,
        config['residual_mode'],
        'relu',
        'relu',
        output_activation,
        budget,
        widths_factory(topology)(num_outputs, depth),
        depth,
        config['topology']
    )

    config['widths'] = widths
    config['network_structure'] = NetworkJSONSerializer(network)()
    print('begin reps: budget: {}, depth: {}, widths: {}, rep: {}'.format(budget, depth, widths, config['rep']))

    ## Create Keras model from NetworkModule
    with strategy.scope():
        keras_inputs, keras_output = make_model_from_network(network)

        assert len(keras_inputs) == 1, 'Wrong number of keras inputs generated'
        keras_input = keras_inputs[0]

        ## Run Keras model on dataset
        run_log = test_network(config, dataset, inputs, outputs, keras_input, keras_output, network, run_loss, widths)

    return run_log


def generate_all_tests_from_config(config: {}):
    """
    Generator yielding all test configs specified by a seed config
    """
    for dataset in config['datasets']:
        config['dataset'] = dataset
        for learning_rate in config['learning_rates']:
            config['optimizer']['config']['learning_rate'] = float(learning_rate)
            for topology in config['topologies']:
                config['topology'] = topology
                for residual_mode in config['residual_modes']:
                    config['residual_mode'] = residual_mode
                    for budget in config['budgets']:
                        config['budget'] = budget

                        if 'epoch_scale' in config:
                            epoch_scale = config['epoch_scale']
                            m = epoch_scale['m']
                            b = epoch_scale['b']
                            epochs = int(numpy.ceil(numpy.exp(m * numpy.log(budget) + b)))
                            config['run_config']['epochs'] = epochs
                            # print(f"budget: {budget}, epochs {epochs}, log {numpy.log(epochs)} m {m} b {b}")

                        for depth in config['depths']:
                            config['depth'] = depth
                            for rep in range(config['reps']):
                                this_config = deepcopy(config)
                                this_config['rep'] = rep
                                this_config['mode'] = 'single'
                                if this_config['seed'] is None:
                                    this_config['seed'] = random.getrandbits(31)
                                yield this_config


def run_aspect_test_from_config(config: {}, strategy=None):
    """
    Entrypoint for the primary use of this module. Take a config that describes many different potential runs, and loop through them in series - logging data after each run.
    """
    log_data = aspect_test(config, strategy=strategy)
    write_log(log_data, config['log'])
    gc.collect()
    print('done.')


# Generate full tests:
# { mode:list, datasets: ['201_pol', '529_pollen', '537_houses', 'adult', 'connect_4', 'mnist', 'nursery', 'sleep', 'wine_quality_white'], 'topologies': ['rectangle', 'trapezoid', 'exponential', 'wide_first'], 'budgets': [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432], 'depths': [2,3,4,5,7,8,9,10,12,14,16,18,20] }

# Modes:
# direct: run multiple aspect tests by iterating through the config and running the test on this machine
# list: create single configs from an aspect-test config. Meant to be piped into a jobqueue enqueue script
# single: run a single test from a config that has already been "unpacked", such as the output of "list". Meant to be used from a jobqueue queue runner.

## Support the python -m runpy interface
if __name__ == "__main__":

    config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
    mode = config['mode']

    strategy = jq_worker.make_strategy(0, 6, 0, 1, 8192)

    if mode == 'single':
        run_aspect_test_from_config(config)
    elif mode == 'direct':
        for this_config in generate_all_tests_from_config(config):
            print(this_config)
            run_aspect_test_from_config(this_config, strategy=strategy)
    elif mode == 'list':
        for this_config in generate_all_tests_from_config(config):
            this_config[
                "jq_module"] = "dmp.experiment.aspect_test"  # Full path to this module. Used by the job queue runner
            json.dump(this_config, sys.stdout)
            print("")  ## newline
    else:
        assert (False), f"Invalid mode {mode}"
