from typing import Iterable, Mapping, Callable

from lmarshal.marshal import Marshal
from lmarshal.types import TypeCode


class Marshaler:
    """
    """

    def __init__(self, marshal: Marshal) -> None:
        self._marshal: Marshal = marshal
        self._vertex_index: {} = {}  # if use_references else None

    def marshal(self, target: any) -> any:
        # marshal target into a plain object
        marshaler_map = self._marshal._marshaler_map
        target_type = type(target)
        if target_type in marshaler_map:
            return marshaler_map[target_type](self, target)
        raise ValueError(f'Type has undefined marshaling protocol: "{target_type}"')

    def marshal_referencable(self, target: any, handler: Callable[['Marshaler', any], any]) -> any:
        target_id = id(target)
        if target_id in self._vertex_index:
            return self.make_reference(target_id)  # if target is already indexed, return its reference index
        self._vertex_index[target_id] = len(self._vertex_index)  # index target
        return handler(self, target)

    def make_reference(self, target_id: int) -> any:
        index = self._vertex_index[target_id]
        return self._marshal._reference_prefix + hex(index)[2:]

    def marshal_passthrough(self, target: any) -> any:
        return target

    def marshal_list(self, target: Iterable) -> any:
        return [self.marshal(e) for e in target]

    def marshal_dict(self, target: Mapping) -> any:
        has_conformant_keys = True
        result = {}
        # marshaled[self._marshal._label_key] = self._vertex_index[id(target)] # TODO: if labeling
        for k, v in sorted(target.items()):
            k, is_conformant = self.marshal_key(k)
            has_conformant_keys = has_conformant_keys and is_conformant
            v = self.marshal(v)
            result[k] = v
        if not has_conformant_keys:  # handle nonconforming dicts
            result = {self._marshal._flat_dict_key: [list(kvp) for kvp in result.items()]}
        return result

    def marshal_key(self, key: any) -> (any, bool):
        is_str = type(key) == str
        if is_str:
            if key in self._marshal._prefix_key_set or key.starts_with(self._marshal._escape_prefix):
                key = self.escape_string(key)
        else:
            key = self.marshal(key)
        return key, is_str

    def marshal_string(self, target: str) -> any:
        if len(target) > 0 and target[0] in self._marshal._prefix_value_set:
            target = self.escape_string(target)
        return target

    def escape_string(self, target: str) -> str:
        return self._marshal._escape_prefix + target

    def marshal_regular_object(self, target: any, type_code: TypeCode) -> any:
        marshaled = self.marshal_dict(vars(target))
        marshaled[self._marshal._type_key] = type_code
        # marshaled[self._marshal._label_key] = self._vertex_index[id(target)] # TODO: if labeling
        return marshaled
