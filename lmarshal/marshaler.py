from typing import Iterable, Mapping, Callable, Optional, MutableMapping

from lmarshal.marshal_config import MarshalConfig
from lmarshal.types import TypeCode


class Marshaler:
    """
    """

    def __init__(self, config: MarshalConfig) -> None:
        self._config: MarshalConfig = config
        self._vertex_index: {int: str} = {}

    def marshal(self, target: any) -> any:
        # marshal target into a plain object
        marshaler_type_map = self._config.marshaler_type_map
        target_type = type(target)
        if target_type in marshaler_type_map:
            return marshaler_type_map[target_type](self, target)
        raise ValueError(f'Type has undefined marshaling protocol: "{target_type}".')

    def marshal_passthrough(self, target: any) -> any:
        return target

    def marshal_string(self, target: str) -> any:
        if len(target) > 0 and target[0] in self._config.control_prefix_set:
            target = self.escape_string(target)
        return target

    def escape_string(self, target: str) -> str:
        return self._config.escape_prefix + target

    def marshal_list(self, target: Iterable) -> any:
        return [self.marshal(e) for e in target]

    def marshal_dict(self, target: Mapping) -> any:
        items = []
        for k, v in target.items():
            if not isinstance(k, str):
                return self.marshal_flat_dict(target)
            if k in self._config.control_prefix_set or k.startswith(self._config.escape_prefix):
                k = self.escape_string(k)
            # TODO: optionally allow referenced strings as keys?
            items.append((k, v))
        return {k: self.marshal(v) for k, v in sorted(items)}

    def marshal_flat_dict(self, target: Mapping):
        return {self._config.flat_dict_key: [[self.marshal(k), self.marshal(v)] for k, v in target.items()]}

    def marshal_key(self, key: any) -> (any, bool):
        is_str = type(key) == str
        if is_str:
            if key in self._config.control_prefix_set or key.startswith(self._config.escape_prefix):
                key = self.escape_string(key)
        else:
            key = self.marshal(key)
        return key, is_str

    def marshal_referencable(
            self,
            target: any,
            handler: Callable[['Marshaler', any], any],
            label: Optional[any] = None,
    ) -> any:
        vertex_index = self._vertex_index
        target_id = id(target)
        if target_id in vertex_index:  # if target is already indexed, return its reference index
            return self.marshal_reference(vertex_index[target_id])

        vertex_index[target_id] = self.make_implicit_label(len(vertex_index)) if label is None else label
        result = handler(self, target)
        if label is not None:
            if not isinstance(result, MutableMapping):
                raise ValueError(
                    f'Label "{label}" specified, but marshaled type "{type(result)}" is not a MutableMapping.')
            result[self._config.label_key] = label  # set label key
        return result

    @staticmethod
    def make_implicit_label(element_number: int) -> str:
        return hex(element_number)[2:]

    def marshal_reference(self, label: str) -> any:
        return self._config.reference_prefix + label

    def make_implicit_reference(self, target_id: int) -> any:
        index = self._vertex_index[target_id]
        return self._config.reference_prefix + hex(index)[2:]

    def marshal_regular_object(self, target: any, type_code: TypeCode) -> any:
        marshaled = self.marshal_dict(vars(target))
        marshaled[self._config.type_key] = type_code
        return marshaled
