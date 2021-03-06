import json
from typing import Dict, List

from dmp.structure.n_add import NAdd
from dmp.structure.n_dense import NDense
from dmp.structure.n_input import NInput
from dmp.structure.network_module import NetworkModule


class NetworkJSONSerializer:
    """
    Serializes a NetworkModule graph into JSON
    """

    type_field_name: str = 'type'
    inputs_field_name: str = 'inputs'

    string_to_type_map: Dict[str, type] = {
        n_cls.__name__: n_cls for n_cls in [
            NInput,
            NDense,
            NAdd,
        ]
    }

    def __init__(self, vertex: NetworkModule) -> None:
        self._vertices: List = []
        self._module_index: Dict[NetworkModule, int] = {}
        self._prepare_serializable(vertex)
        self._serialized: str = json.dumps(self._vertices, sort_keys=True, separators=(',', ':'))

    def __call__(self) -> str:
        return self._serialized

    def _prepare_serializable(self, target: NetworkModule) -> int:
        if target in self._module_index:
            return self._module_index[target]

        serialized = vars(target).copy()

        index = len(self._vertices)
        self._vertices.append(serialized)
        self._module_index[target] = index

        serialized[NetworkJSONSerializer.inputs_field_name] = \
            [self._prepare_serializable(input) for input in target.inputs]

        assert(NetworkJSONSerializer.type_field_name not in serialized)
        serialized[NetworkJSONSerializer.type_field_name] = type(target).__name__
        return index
