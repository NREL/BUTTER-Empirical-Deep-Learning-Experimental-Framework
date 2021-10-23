from typing import Iterable, Mapping, Callable, Optional, MutableMapping

from lmarshal.marshal_config import MarshalConfig
from lmarshal.types import TypeCode


class Marshaler:
    """
    """

    def __init__(self, config: MarshalConfig, source: any) -> None:
        self._config: MarshalConfig = config
        self._vertex_index: {int: str} = {}
        self._result: any = self.marshal(source)

    def __call__(self) -> any:
        return self._result

    def marshal(self, source: any) -> any:
        # marshal source into a plain object
        marshaler_type_map = self._config.marshaler_type_map
        source_type = type(source)
        if source_type in marshaler_type_map:
            return marshaler_type_map[source_type](self, source)
        raise ValueError(f'Type has undefined marshaling protocol: "{source_type}".')

    def marshal_passthrough(self, source: any) -> any:
        return source

    def marshal_string(self, source: str) -> any:
        if len(source) > 0 and source[0] in self._config.control_prefix_set:
            source = self.escape_string(source)
        return source

    def escape_string(self, source: str) -> str:
        return self._config.escape_prefix + source

    def marshal_list(self, source: Iterable) -> any:
        return [self.marshal(e) for e in source]

    def marshal_dict(self, source: Mapping) -> any:
        items = []
        config = self._config
        for k, v in source.items():
            if not isinstance(k, str):
                return {self._config.flat_dict_key: [[self.marshal(k), self.marshal(v)] for k, v in source.items()]}
            if k in config.control_key_set or k.startswith(config.escape_prefix):
                k = self.escape_string(k)
            # TODO: optionally allow referenced strings as keys?
            items.append((k, v))
        return {k: self.marshal(v) for k, v in sorted(items)}

    def marshal_referencable(
            self,
            source: any,
            handler: Callable[['Marshaler', any], any],
            label: Optional[any] = None,
    ) -> any:
        vertex_index = self._vertex_index
        source_id = id(source)
        if source_id in vertex_index:  # if source is already indexed, return its reference index
            return self.marshal_reference(vertex_index[source_id])

        vertex_index[source_id] = self._config.make_implicit_label(len(vertex_index)) if label is None else label
        result = handler(self, source)
        if label is not None:
            if not isinstance(result, MutableMapping):
                raise ValueError(
                    f'Label "{label}" specified, but marshaled type "{type(result)}" is not a MutableMapping.')
            result[self._config.label_key] = label  # set label key
        return result

    def marshal_reference(self, label: str) -> any:
        return self._config.reference_prefix + label

    def make_implicit_reference(self, source_id: int) -> any:
        index = self._vertex_index[source_id]
        return self._config.reference_prefix + hex(index)[2:]

    def marshal_regular_object(self, source: any, type_code: TypeCode) -> any:
        marshaled = self.marshal_dict(vars(source))
        marshaled[self._config.type_key] = type_code
        return marshaled
