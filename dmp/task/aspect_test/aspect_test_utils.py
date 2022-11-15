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
from cnn.cnn_net import get_cell


def add_label_noise(label_noise, run_task, train_outputs):
    if label_noise is not None and label_noise != 'none' and label_noise != 0.0:
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


def count_vars_in_keras_model(model: Model, var_getter) -> int:
    count = 0
    for var in var_getter(model):
        acc = 1
        for dim in var.get_shape():
            acc *= int(dim)
        count += acc
    return count


def count_trainable_parameters_in_keras_model(model: Model) -> int:
    return count_vars_in_keras_model(model, lambda m: m.trainable_variables)


def count_parameters_in_keras_model(model: Model) -> int:
    return count_vars_in_keras_model(model, lambda m: m.variables)


def count_non_trainable_parameters_in_keras_model(model: Model) -> int:
    return count_vars_in_keras_model(model, lambda m: m.non_trainable_variables)

# TODO: clean up this visitor
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
        def visit(self, target) -> Any:
            raise NotImplementedError(
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

        # CNN registers
        @visit.register
        def _(self, target: NCNNInput) -> int:
            return 0

        @visit.register
        def _(self, target: NConv) -> int:
            params = target.kernel_size ** 2 * target.channels
            for i in target.inputs:
                params *= i.channels
            return params

        @visit.register
        def _(self, target: NSepConv) -> int:
            params = 2 * target.kernel_size * target.channels
            for i in target.inputs:
                params *= i.channels
            return params

        @visit.register
        def _(self, target: NMaxPool) -> int:
            return 0

        @visit.register
        def _(self, target: NGlobalPool) -> int:
            return 0

        @visit.register
        def _(self, target: NIdentity) -> int:
            return 0

        @visit.register
        def _(self, target: NZeroize) -> int:
            return 0
        
        @visit.register
        def _(self, target: NConcat) -> int:
            return 0

        # CNN Cell Registers
        @visit.register
        def _(self, target: NConvStem) -> int:
            return 9 * target.channels * target.input_channels

        @visit.register
        def _(self, target: NCell) -> int:
            if target.cell_type == 'parallelconcat':
                channels = [target.channels//target.nodes for _ in range(target.nodes)]
                for i in range(target.channels % target.nodes):
                    channels[i] += 1
            else:
                channels = [target.channels for _ in range(target.nodes)]
            params = 0 
            in_channels = target.inputs[0].channels
            params_dict = {'conv3x3': 9, 'conv5x5': 25, 'sepconv3x3': 6, 'sepconv5x5': 10,
                'conv1x1': 1, 'maxpool3x3': 0, 'avgpool3x3': 0, 'identity': 0,
                'zeroize': 0, 'projection': 0}
            for i in range(target.nodes):
                num_channels = channels[i]
                ops = target.operations[i]
                for j in range(len(ops)):
                    op = ops[j]
                    if j > 0:
                        params += params_dict[op] * num_channels**2 
                    else:
                        params += params_dict[op] * num_channels * in_channels
            return params

        @visit.register
        def _(self, target: NDownsample) -> int:
            params = 1
            for i in target.inputs:
                params *= i.channels
            params *= target.channels
            return params

        @visit.register
        def _(self, target: NFinalClassifier) -> int:
            params = 1
            for i in target.inputs:
                params *= i.channels
            params *= target.classes 
            return params

    return sum((GetNumFreeParameters().visit(i) for i in build_set_of_modules(target)))


def make_network(
        input_shape: Tuple[int, ...],
        widths: List[int],
        residual_mode: Optional[str],
        input_activation: str,
        internal_activation: str,
        output_activation: str,
        layer_args: dict,
) -> NetworkModule:
    # print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))

    input_layer = NInput(label=0,
                         inputs=[],
                         shape=list(input_shape[1:]))
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
            # If this isn't the first or last layer, and the previous layer is
            # of the same width insert a residual sum between layers
            # NB: Only works for rectangle
            if d > 0 and d < len(widths)-1 and layer_width == widths[d-1]:
                layer = NAdd(label=0,
                             inputs=[layer, current],
                             shape=layer.shape.copy())
        else:
            raise NotImplementedError(
                f'Unknown residual mode "{residual_mode}".')
        current = layer
    return current

def get_params_and_type_from_config(
    config:dict, 
    type_key:str = 'type',
    ) -> Tuple[str, dict]:
    params = config.copy()
    del params['type']
    return config['type'], params

def make_from_config(
    config:dict,
    mapping:Dict[str, Callable],
    config_name : str,
    *args,
    **kwargs,
) -> Any:
    type, params = get_params_and_type_from_config(config)
    factory = mapping.get(type, None)
    if factory is None:
        raise NotImplementedError(f'Unknown {config_name} type "{type}".')
    return factory(*args, **kwargs, **params)

def make_regularizer(config: Optional[Dict]) \
        -> Optional[keras.regularizers.Regularizer]:
    if config is None:
        return None
    return make_from_config(
        config,
        {
            'l1':keras.regularizers.L1,
            'l2':keras.regularizers.L2,
            'l1l2':keras.regularizers.L1L2
        },
        'regularizer'
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
    cell_depths = [cell_depth//(downsamples+1) for _ in range(downsamples+1)]
    for i in range(cell_depth%(downsamples+1)):
        cell_depths[-i-1] += 1

    for i in range(downsamples+1):
        # Add downsampling layer
        if i > 0:
            layer_list.append('downsample')
            widths_list.append(widths[i])
        # Add cells
        for _ in range(cell_depths[i]):
            layer_list.append('cell')
            widths_list.append(widths[i])

    input_layer = NInput(label=0,
                shape=input_shape,)
    if not cell_setup: # do the updated keras layer wise construction
        current = generate_conv_stem(input_layer, widths[0], batch_norm)
        # Loop through layers 
        for i in range(len(layer_list)):
            layer_width = widths_list[i]
            layer_type = layer_list[i]
            if layer_type == 'cell':
                layer = generate_generic_cell(type=cell_type, inputs=current, nodes=cell_nodes,
                    channels=layer_width, operations=cell_ops, batch_norm=batch_norm, activation=internal_activation)
            elif layer_type == 'downsample':
                layer = generate_downsample(inputs=current, channels=layer_width, 
                                batch_norm=batch_norm, activation=internal_activation)
            else:
                raise ValueError(f'Unknown layer type {layer_type}')
            current = layer
        # Add final classifier
        final_classifier = generate_final_classifier(inputs=current, classes=classes,
                                                    activation=output_activation)
    else: # Do the outdated cell-wise construction
        current = NConvStem(
            inputs=[input_layer, ],
            activation=internal_activation,
            channels=widths[0],
            batch_norm=batch_norm,
            input_channels=input_shape[-1],
        )
        # Loop through layers 
        for i in range(len(layer_list)):
            layer_width = widths_list[i]
            layer_type = layer_list[i]
            if layer_type == 'cell':
                layer = NCell(
                    inputs=[current, ],
                    activation=internal_activation,
                    channels=layer_width,
                    batch_norm=batch_norm,
                    cell_type=cell_type,
                    nodes=cell_nodes,
                    operations=cell_ops,
                )
            elif layer_type == 'downsample':
                layer = NDownsample(
                    inputs=[current, ],
                    activation=internal_activation,
                    channels=layer_width,
                )
            else:
                raise ValueError(f'Unknown layer type {layer_type}')
            current = layer

        # Add final classifier
        final_classifier = NFinalClassifier(
            inputs=[current, ],
            activation=output_activation,
            classes=classes,
        )
    return final_classifier





class MakeKerasLayersFromNetwork:
    """
    Visitor that makes a keras module for a given network module and input keras modules
    """

    def __init__(self, target: NetworkModule) -> None:
        self._inputs: list = []
        self._node_layer_map: Dict[NetworkModule, tensorflow.keras.Layer] = {}
        self._outputs: list = []

        output = self._visit(target)
        self._outputs = [output]

    def __call__(self) -> Tuple[
        list,
        list,
        Dict[NetworkModule, tensorflow.keras.Layer],
    ]:
        return self._inputs, self._outputs, self._node_layer_map

    def _visit(self, target: NetworkModule) \
            -> Tuple[Any, List[Any], Dict[NetworkModule, Any], List[Any]]:
        if target not in self._node_layer_map:
            keras_inputs = [self._visit(i) for i in target.inputs]
            keras_layer = self._visit_raw(target, keras_inputs)
            self._node_layer_map[target] = keras_layer
        return self._node_layer_map[target]

    @singledispatchmethod
    def _visit_raw(self, target, keras_inputs) -> Any:
        raise NotImplementedError(
            'Unsupported module of type "{}".'.format(type(target)))

    @_visit_raw.register
    def _(self, target: NInput, keras_inputs) -> Any:
        result = Input(shape=target.shape)
        self._inputs.append(result)
        return result

    @_visit_raw.register
    def _(self, target: NDense, keras_inputs) -> Any:

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
            kernel_initializer=target.kernel_initializer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NAdd, keras_inputs) -> Any:
        return tensorflow.keras.layers.add(keras_inputs)

    @_visit_raw.register
    def _(self, target: NConcat, keras_inputs) -> any:
        return keras.layers.Concatenate()(keras_inputs)

    # CNN visitors 
    @_visit_raw.register
    # def _(self, target: NCNNInput, keras_inputs) -> any:
    #     result = Input(shape=target.shape)
    #     self._inputs.append(result)
    #     return result

    @_visit_raw.register
    def _(self, target: NConv, keras_inputs) -> any:
        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        if target.kernel_size == 3:
            cell = Conv3x3Operation 
        elif target.kernel_size == 5:
            cell = Conv5x5Operation
        elif target.kernel_size == 1:
            cell = Conv1x1Operation
        
        return cell(
            target.channels,
            activation=target.activation,
            batch_norm=target.batch_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NSepConv, keras_inputs) -> any:
        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        if target.kernel_size == 3:
            cell = SepConv3x3Operation 
        elif target.kernel_size == 5:
            cell = SepConv5x5Operation
        
        return cell(
            target.channels,
            activation=target.activation,
            batch_norm=target.batch_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NMaxPool, keras_inputs) -> any:
        return keras.layers.MaxPool2D(pool_size=target.kernel_size, 
            strides=target.stride, 
            padding=target.padding)(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NGlobalPool, keras_inputs) -> any:
        return keras.layers.GlobalAveragePooling2D()(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NIdentity, keras_inputs) -> any:
        return IdentityOperation()(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NZeroize, keras_inputs) -> any:
        return ZeroizeOperation()(*keras_inputs)

    #CNN Cell Registers
    @_visit_raw.register
    def _(self, target: NCell, keras_inputs) -> any:
        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        return get_cell(type=target.cell_type,
            nodes=target.nodes,
            operations=target.operations,
            channels=target.channels,
            batch_norm=target.batch_norm,
            activation=target.activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NConvStem, keras_inputs) -> any:
        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)
        print(keras_inputs)

        return ConvStem(
            target.channels,
            activation=target.activation,
            batch_norm=target.batch_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs) # Expands a list as though it were separate arguments

    @_visit_raw.register
    def _(self, target: NDownsample, keras_inputs) -> any:
        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        return DownsampleCell(
            target.channels,
            activation=target.activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NFinalClassifier, keras_inputs) -> any:
        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        return FinalClassifier(
            target.classes,
            activation=target.activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )(*keras_inputs)


def make_keras_network_from_network_module(target: NetworkModule) \
        -> Tuple[keras.Model, Dict[NetworkModule, tensorflow.keras.Layer] ]:
    """
    Recursively builds a keras network from the given network module and its directed acyclic graph of inputs
    :param target: starting point module
    :param model_cache: optional, used internally to preserve information through recursive calls
    """
    inputs, outputs, node_layer_map = MakeKerasLayersFromNetwork(target)()
    return Model(inputs=inputs, outputs=outputs), node_layer_map


def compute_network_configuration(num_outputs, ml_task: str) -> Tuple[Any, Any]:
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
    input_shape: Tuple[int, ...],
    residual_mode: Optional[str],
    input_activation: str,
    internal_activation: str,
    output_activation: str,
    target_size: int,
    make_widths: Callable[[int], List[int]],
    layer_args: dict
) -> Tuple[int, List[int], NetworkModule]:
    best = (math.inf, None, None)

    def search_objective(w0):
        nonlocal best
        widths = make_widths(w0)
        network = make_network(input_shape,
                               widths,
                               residual_mode,
                               input_activation,
                               internal_activation,
                               output_activation,
                               layer_args,
                               )
        delta = count_num_free_parameters(network) - target_size

        if abs(delta) < abs(best[0]):
            best = (delta, widths, network)

        return delta

    binary_search_int(search_objective, 1, int(2 ** 30))
    return best  # type: ignore


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
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 4)


def get_wide_first_8x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 8)


def get_wide_first_16x(num_outputs: int, depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(num_outputs, depth, 16)


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
