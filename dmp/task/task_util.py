from typing import Dict, Iterable, Tuple


def remap_key_prefixes(target: Dict,
                       prefix_mapping: Iterable[Tuple[str, str]]) -> dict:
    result = {}
    for k, v in target.items():
        if isinstance(k, str):
            for from_prefix, to_prefix in prefix_mapping:
                if k.startswith(from_prefix):
                    k = to_prefix + k[len(from_prefix):]
                    break
        result[k] = v
    return result
