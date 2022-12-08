import math
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from dmp.layer.conv_cell import make_graph_cell, make_parallel_add_cell
from dmp.layer.visitor.compute_layer_shapes import compute_layer_shapes
from dmp.layer.visitor.count_free_parameters import count_free_parameters
from dmp.layer import *
from dmp.model.network_info import NetworkInfo

K = TypeVar('K')
V = TypeVar('V')


def dispatch(
    key: K,  # key to dispatch on
    dispatch_table: Dict[K, V],
    dispatch_name: str,
) -> V:
    try:
        return dispatch_table[key]
    except KeyError:
        raise NotImplementedError(f'Unknown {dispatch_name}, "{key}".')


def make_dispatcher(
    dispatch_name: str,  # used when raising an exception
    dispatch_table: Dict[K, V],
) -> Callable[[K], V]:

    def dispatch_function(name: K) -> V:
        return dispatch(name, dispatch_table, dispatch_name)

    return dispatch_function


type_key: str = 'type'


def get_params_and_type_from_config(config: dict) -> Tuple[str, dict]:
    params = config.copy()
    del params[type_key]
    return config[type_key], params


def make_from_config_using_keras_get(
        config: dict,
        keras_get_function: Callable,
        name: str,  # used for exception messages
) -> Any:
    if config is None:
        return None

    type, params = get_params_and_type_from_config(config)
    result = keras_get_function({'class_name': type, 'config': params})
    if result is None:
        raise ValueError(f'Unknown {name}, {config}.')


def make_typed_config_factory(
        name: str,  # name of the thing we are making from the config
        type_dispatch_table: Dict[str, Callable],  # factory dispatch table 
) -> Callable:

    dispatch_function = make_dispatcher(name, type_dispatch_table)

    def factory(
        config: Optional[Dict],  # config to use with type key
        *args,  # forwarded args
        **kwargs,
    ):
        if config is None:
            return None

        type, params = get_params_and_type_from_config(config)
        return dispatch_function(type)(*args, **kwargs, **params)

    return factory


def make_from_optional_typed_config(
    config: dict,
    key: str,  # name of the thing we are making from the config
    dispatch_factory: Callable,
) -> Any:
    return dispatch_factory(config.get(key, None))


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
    num_outputs: int,
    ml_task: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output_activation = 'relu'
    if ml_task == 'regression':
        run_loss = 'MeanSquaredError'
        output_activation = 'sigmoid'
    elif ml_task == 'classification':
        if num_outputs == 1:
            output_activation = 'sigmoid'
            run_loss = 'BinaryCrossentropy'
        else:
            output_activation = 'softmax'
            run_loss = 'CategoricalCrossentropy'
    else:
        raise Exception('Unknown task "{}"'.format(ml_task))

    return {type_key: output_activation}, {type_key: run_loss}


def binary_search_int(
    objective: Callable[[int], Union[int, float]],
    minimum: int,
    maximum: int,
) -> Tuple[int, bool]:
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


def binary_search_float(
    objective: Callable[[float], Union[int, float]],
    minimum: float,
    maximum: float,
    epsilon: float,
) -> Tuple[float, bool]:
    candidate = 0.0
    while minimum < maximum:
        candidate = (maximum + minimum) / 2
        evaluation = objective(candidate)

        if evaluation < 0:  # candidate < target
            minimum = candidate + epsilon
        elif evaluation > 0:  # candidate > target
            maximum = candidate - epsilon
        else:  # candidate == target
            return candidate, True
    return candidate, False


T = TypeVar('T')


def _find_closest_network_to_target_size(
    target_num_free_parameters: int,
    make_network: Callable[[T], NetworkInfo],
    search_function: Callable,
) -> Tuple[T, NetworkInfo]:
    best = (math.inf, None, None, {})

    def search_objective(search_parameter):
        nonlocal best
        network = make_network(search_parameter)
        delta = network.num_free_parameters - target_num_free_parameters
        if abs(delta) < abs(best[0]):
            best = (delta, network)
        return delta

    search_function(search_objective)
    return best  # type: ignore


def find_closest_network_to_target_size_float(
    target_num_free_parameters: int,
    make_network: Callable[[float], NetworkInfo],
) -> Tuple[int, NetworkInfo]:
    return _find_closest_network_to_target_size(
        target_num_free_parameters,
        make_network,
        lambda search_objective: binary_search_float(
            search_objective,
            1.0,
            float(2**31),
            1e-6,
        ),
    )


def find_closest_network_to_target_size_int(
    target_num_free_parameters: int,
    make_network: Callable[[float], NetworkInfo],
) -> Tuple[int, NetworkInfo]:
    return _find_closest_network_to_target_size(
        target_num_free_parameters,
        make_network,
        lambda search_objective: binary_search_int(
            search_objective,
            1,
            int(2**30),
        ),
    )


def remap_key_prefixes(
    target: Dict,
    prefix_mapping: Iterable[Tuple[str, str]],
) -> dict:
    result = {}
    for k, v in target.items():
        if isinstance(k, str):
            for from_prefix, to_prefix in prefix_mapping:
                if k.startswith(from_prefix):
                    k = to_prefix + k[len(from_prefix):]
                    break
        result[k] = v
    return result


def flatten(items):
    '''
    Generator that recursively flattens an Iterable
    '''
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
