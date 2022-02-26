import json

from dmp.structure.algorithm.network_json_serializer import NetworkJSONSerializer
from dmp.structure.network_module import NetworkModule


class NetworkJSONDeserializer:
    """
    Deserializes a NetworkModule graph into JSON
    """

    def __init__(self, serialized: str) -> None:
        raw_vertices: [{}] = json.loads(serialized)
        vertices: [NetworkModule] = [self._make_module(raw_vertex) for raw_vertex in raw_vertices]
        for vertex in vertices:
            vertex.inputs = [vertices[index] for index in vertex.inputs]
        self._vertices: [NetworkModule] = vertices

    def __call__(self) -> [NetworkModule]:
        return self._vertices[0]

    def _make_module(self, raw_vertex: {}) -> int:
        vertex_typename = raw_vertex[NetworkJSONSerializer.type_field_name]
        vertex_type = NetworkJSONSerializer.string_to_type_map[vertex_typename]
        del raw_vertex[NetworkJSONSerializer.type_field_name]
        return vertex_type(**raw_vertex)
