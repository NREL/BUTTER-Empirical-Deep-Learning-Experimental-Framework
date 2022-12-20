import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union
from dmp.common import binary_search_float, binary_search_int
from dmp.model.network_info import NetworkInfo

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

