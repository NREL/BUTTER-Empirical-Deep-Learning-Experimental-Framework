import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

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
