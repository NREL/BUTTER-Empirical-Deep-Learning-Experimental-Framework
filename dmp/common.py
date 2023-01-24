from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union
import math

tensorflow_type_key: str = 'class_name'
tensorflow_config_key: str = 'config'
keras_type_key: str = 'class'
marshal_type_key: str = 'type'

K = TypeVar('K')
V = TypeVar('V')


def dispatch(
        dispatch_name: str,
        dispatch_table: Dict[K, V],
        key: K,  # key to dispatch on
) -> V:
    try:
        return dispatch_table[key]
    except KeyError:
        raise NotImplementedError(f'Unknown {dispatch_name}, "{key}".')


def make_dispatcher(
    dispatch_name: str,  # used when raising an exception
    dispatch_table: Dict[K, V],
) -> Callable[[K], V]:

    def dispatch_function(key: K) -> V:
        return dispatch(dispatch_name, dispatch_table, key)

    return dispatch_function


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


def flatten(items, levels: int = -1):
    '''
    Generator that recursively flattens nested Iterables
    '''
    for x in items:
        if levels != 0 and isinstance(
                x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x, levels=levels - 1)
        else:
            yield x


def flatten_dict(items: Mapping, connector: str):
    '''
    Generator that recursively flattens dicts
    '''

    def do_flatten(prefix, target):
        if isinstance(target, Mapping):
            for k, v in target.items():
                yield from do_flatten(prefix + connector + k, v)
        else:
            yield (prefix, target)

    yield from do_flatten('', items)
