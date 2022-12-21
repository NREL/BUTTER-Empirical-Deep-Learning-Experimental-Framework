from typing import Any, Dict, Iterable, Mapping, Sequence, Set, Tuple, Type

from .common_marshaler import CommonMarshaler
from .marshal_config import MarshalConfig
from .marshal_types import RawObjectMarshaler, TypeCode


class Marshaler(CommonMarshaler):
    __slots__ = [
        '_reference_index',
        '_type_map',
        '_vertex_index',
        '_string_index',
        '_referenced',
        '_result',
    ]

    def __init__(
        self,
        config: MarshalConfig,
        type_map: Dict[Type, RawObjectMarshaler],
        source: Any,
    ) -> None:
        super().__init__(config)
        self._type_map: Dict[Type, RawObjectMarshaler] = type_map
        self._vertex_index: Dict[int, Tuple[str, Any, Any]] = {}
        self._string_index: Dict[str, str] = {}
        self._referenced: Set[int] = set()
        self._result: Any = self.marshal(source)
        if self._config.label_referenced and not self._config.label_all:
            for element_id in self._referenced:
                label, _, element = self._vertex_index[element_id]
                if type(element) is dict:
                    element[self._config.label_key] = label

    def __call__(self) -> Any:
        return self._result

    def marshal(self, source: Any) -> Any:
        # marshal source into a plain object
        source_type = type(source)
        if source_type not in self._type_map:
            raise TypeError(
                f'Type has undefined marshaling protocol: "{source_type}".')
        return self._type_map[source_type](self, source)

    @staticmethod
    def marshal_passthrough(marshaler: 'Marshaler', source: Any) -> Any:
        return source

    @staticmethod
    def marshal_untyped(
        marshaler: 'Marshaler',
        source: Any,
        object_marshaler: RawObjectMarshaler,
    ) -> Any:
        vertex_index = marshaler._vertex_index
        source_id = id(source)
        if source_id in vertex_index:  # if source is already indexed, return its reference index
            label, _, dest = vertex_index[source_id]
            if marshaler._config.label_referenced:
                marshaler._referenced.add(source_id)
            if marshaler._config.circular_references_only and dest is not None:
                return dest
            return marshaler._config.reference_prefix + label

        label = marshaler._make_label(len(vertex_index))
        vertex_index[source_id] = (label, source, None)
        dest = object_marshaler(marshaler, source)
        vertex_index[source_id] = (label, source, dest)

        if marshaler._config.label_all and type(dest) is dict:
            dest[marshaler._config.label_key] = label
        return dest

    @staticmethod
    def marshal_string(marshaler: 'Marshaler', source: str) -> str:
        if len(source
               ) > 0 and source[0] in marshaler._config.control_prefix_set:
            source = marshaler._escape_string(source)
        return source

    @staticmethod
    def canonicalize_and_marshal_string(m: 'Marshaler', s: str):
        if s in m._string_index:
            s = m._string_index[s]
        else:
            m._string_index[s] = s
        return Marshaler.marshal_untyped(m, s, Marshaler.marshal_string)

    @staticmethod
    def marshal_list(marshaler: 'Marshaler', source: Iterable) -> list:
        return Marshaler.marshal_untyped(
            marshaler, source, lambda m, s: [m.marshal(e) for e in s])

    @staticmethod
    def marshal_dict_items(m: 'Marshaler', s: Iterable[Sequence]) -> dict:
        use_flat_dict = False
        items = []
        for t in s:
            k = t[0]
            if type(k) is not str:
                use_flat_dict = True
            else:
                t = (m.marshal_key(k), t[1])
            items.append(t)

        if use_flat_dict:
            return {
                m._config.flat_dict_key: [[
                    m.marshal(m.demarshal_key(k) if type(k) is str else k),
                    m.marshal(v)
                ] for k, v in items]
            }

        items.sort()
        result = {k: m.marshal(v) for k, v in items}
        return result

    @staticmethod
    def marshal_bare_dict(m: 'Marshaler', s: Mapping) -> dict:
        return Marshaler.marshal_dict_items(m, s.items())

    @staticmethod
    def marshal_dict(marshaler: 'Marshaler', source: Mapping) -> dict:
        return Marshaler.marshal_untyped(marshaler, source,
                                         Marshaler.marshal_bare_dict)

    @staticmethod
    def marshal_typed(
        marshaler: 'Marshaler',
        source: Any,
        type_code: TypeCode,
        object_marshaler: RawObjectMarshaler,
    ) -> Any:

        def type_checked_object_marshaler(m: Marshaler, s: Any) -> dict:
            result = object_marshaler(m, s)
            if type(result) is not dict:
                raise TypeError(
                    f'ObjectMarshaler for type {type(source)} returned a {type(result)} instead of a dict.'
                )
            result[m._config.type_key] = type_code
            return result

        return marshaler.marshal_untyped(marshaler, source,
                                         type_checked_object_marshaler)

    @staticmethod
    def default_object_marshaler(
        marshaler: 'Marshaler',
        source: Any,
    ) -> Dict:
        return Marshaler.marshal_bare_dict(marshaler, vars(source))

    @staticmethod
    def custom_marshalable_marshaler(
        marshaler: 'Marshaler',
        source: Any,
    ) -> Dict:
        return Marshaler.marshal_bare_dict(marshaler, source.marshal())

    @staticmethod
    def enum_marshaler(
        marshaler: 'Marshaler',
        source: Any,
    ) -> Dict:
        return {
            marshaler.marshal_key(marshaler._config.enum_value_key):
            marshaler.marshal(source.value)
        }

    @staticmethod
    def initialize_type_map(type_map: Dict[Type, RawObjectMarshaler],
                            config: MarshalConfig) -> None:
        type_map[type(None)] = Marshaler.marshal_passthrough
        type_map[bool] = Marshaler.marshal_passthrough
        type_map[int] = Marshaler.marshal_passthrough
        type_map[float] = Marshaler.marshal_passthrough

        if config.reference_strings:
            type_map[str] = Marshaler.canonicalize_and_marshal_string
        else:
            type_map[str] = Marshaler.marshal_string

        type_map[list] = Marshaler.marshal_list
        type_map[dict] = Marshaler.marshal_dict
