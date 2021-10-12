from dataclasses import dataclass
from typing import Callable, Optional, Hashable, Type, Union, Sequence, Set, Mapping

TypeCode = Hashable
AtomicData = Union[int, float, str]
MarshaledData = Union[AtomicData, dict, list]

ObjectMarshaler = Optional[Callable[[any], MarshaledData]]
ObjectDemarshaler = Callable[[{}], any]


@dataclass
class MarshallingInfo:
    __slots__ = ('marshaler', 'demarshaler', 'recursive_members')
    marshaler: ObjectMarshaler
    demarshaler: ObjectDemarshaler
    recursive_members: Optional[Set[AtomicData]]


class ObjectTranslator:
    """
    internal map of object type to deserializer
        and object to serializer

    serializer:
        + switch on type
        + vars = vars(target).copy() (default)
        + vars[type] = type (always)
        + special function to handle graph fields
    default deserializer:
        + delete vars[type]
        + switch on type
        + type(**vars)
        + special function to handle graph fields
    """

    def __init__(
            self,
            type_field: str = '__type',
    ) -> None:
        self._type_field: str = type_field
        self._passthrough_types: {Type} = {type(None), int, float}
        self._demarshaler_map: {TypeCode: MarshallingInfo} = {}
        self._marshaler_map: {Type: MarshallingInfo} = {}

        self.depth: int = 0
        self.vertex_index: {int, int} = {}
        # self.vertices: [PlainData] = []
        # self.object_map: {int, any} = {}
        self.object_map: [any] = []
        self.recursive_fields: [] = []

        self.register_passthrough_type(str)
        self.register_passthrough_type(list)
        self.register_passthrough_type(dict)

    def register_type(
            self,
            target_type: Type,
            demarshaler: Optional[ObjectDemarshaler] = None,
            marshaler: Optional[ObjectMarshaler] = None,
            recursive_members: Optional[Sequence[AtomicData]] = None,
            type_code: Optional[AtomicData] = None,
    ) -> None:
        type_code = target_type.__name__ if type_code is None else type_code
        marshaler = self._make_default_marshaler(target_type) if marshaler is None else marshaler
        demarshaler = self._make_default_demarshaler(target_type) if demarshaler is None else demarshaler
        recursive_members = set() if recursive_members is None else {m for m in recursive_members}
        marshalling_info = MarshallingInfo(marshaler, demarshaler, recursive_members)
        self._marshaler_map[target_type] = marshalling_info
        self._demarshaler_map[type_code] = marshalling_info

    def register_passthrough_type(self, target_type: Type) -> None:
        self.register_type(target_type, self._passthrough, self._passthrough)

    def marshal(self, source: any, using_references: bool = True) -> MarshaledData:
        passthrough_types = self._passthrough_types
        marshaler_map = self._marshaler_map
        vertex_index = {}
        vertex_visited_set = set()

        def do_marshal(target: any, use_references: bool) -> MarshaledData:
            target_type = type(target)

            if target_type in passthrough_types:
                return target

            # marshal target into a plain object
            try:
                conversion_info: MarshallingInfo = marshaler_map[target_type]
            except KeyError:
                raise ValueError(f'Unknown target type "{target_type}"')
            marshalled = conversion_info.marshaler(target)

            if type(marshalled) in passthrough_types:
                return marshalled  # if marshalled version is passthrough, return it

            target_id = id(target)
            if use_references:
                if target_id in vertex_index:  # if target is already indexed, return its reference index
                    return vertex_index[target_id]
            elif target_id in vertex_visited_set:  # detect cycles when not using references
                raise ValueError('Circular reference detected')

            vertex_index[target_id] = len(vertex_index)  # index target

            if not isinstance(marshalled, str):  # done for strings
                return marshalled

            # recur into Sequence and Mapping results
            is_sequence = isinstance(marshalled, Sequence)
            if not is_sequence and not isinstance(marshalled, Mapping):
                raise ValueError(f'Invalid marshalled type "{target_type}"')

            vertex_visited_set.add(target_id)  # add to cycle detection set
            if is_sequence:
                marshalled = [do_marshal(e, use_references) for e in marshalled]
            else:
                marshalled = {k: do_marshal(v, use_references and k in conversion_info)
                              for k, v in sorted(marshalled.items())}
            vertex_visited_set.remove(target_id)  # remove from cycle detection set

            return marshalled

        return do_marshal(source, using_references)

    def demarshal(self, source: MarshaledData, using_references: bool = True) -> any:
        passthrough_types = self._passthrough_types
        demarshaler_map = self._demarshaler_map
        vertex_index = []

        def do_demarshal(target: any, use_references: bool) -> MarshaledData:
            target_type = type(target)

            if use_references and isinstance(target_type, int):
                # demarshal references
                demarshaled = None
                if len(vertex_index) <= target:
                    demarshaled = vertex_index[target]
                if demarshaled is None:
                    raise ValueError(f'Undefined reference {target}')
                return demarshaled
            elif target_type in passthrough_types or isinstance(target_type, str):
                return target  # fast-bypass for standard atomic types and any str
            elif isinstance(target_type, Sequence):
                pass
            elif isinstance(target_type, Mapping):
                pass
            else:
                return target  # atomic types pass through and undefined behavior for non-marshaled data types

            target_index = len(vertex_index)
            vertex_index.append(None)  # reserve index slot

            # demarshal target
            if isinstance(target_type, )
                if isinstance(target_type, list):
                    pass
            conversion_info: MarshallingInfo = demarshaler_map[type_code]
            plain = conversion_info.marshaler(target)

            # recurse into lists and objects
            if type(plain) is list:
                plain = [do_demarshal(e, use_references) for e in plain]
            elif type(plain) is dict:
                plain = {k: do_demarshal(plain[k], k in conversion_info) for k, v in plain}

            return plain

        return do_demarshal(source, using_references)

    @staticmethod
    def _make_default_type_code(type: Type) -> TypeCode:
        return type.__name__

    @staticmethod
    def _make_default_demarshaler(type: Type) -> ObjectDemarshaler:
        return lambda translator, source: type(**source)

    @staticmethod
    def _make_default_marshaler(type: Type) -> ObjectMarshaler:
        return lambda translator, source: vars(source)

    @staticmethod
    def _passthrough(target):
        return target

    # def register_type_code(
    #         self,
    #         type_code: TypeCode,
    #         demarshaler: ObjectDemarshaler = None,
    #         marshaler: ObjectMarshaler = None,
    # ) -> None:
    #     self._demarshaler_map[type_code] = demarshaler
    #     self._marshaler_map[type_code] = marshaler
    #
    # def register_type(
    #         self,
    #         type: Type,
    #         type_code: Optional[TypeCode] = None,
    #         demarshaler: Optional[ObjectDemarshaler] = None,
    #         marshaler: Optional[ObjectMarshaler] = None,
    # ) -> None:
    #     marshaler = self._make_default_marshaler(type) if marshaler is None else marshaler
    #     self._marshaler_type_map[type] = marshaler
    #     demarshaler = self._make_default_demarshaler(type) if demarshaler is None else demarshaler
    #     type_code = self._make_default_type_code(type) if type_code is None else type_code
    #     return self.register_type_code(type_code, demarshaler, marshaler)
