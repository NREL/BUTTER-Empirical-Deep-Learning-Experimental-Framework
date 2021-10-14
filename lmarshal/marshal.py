from dataclasses import dataclass
from types import NoneType
from typing import Optional, Type, Sequence, Set, Mapping

from lmarshal.marshaler import Marshaler
from lmarshal.types import ObjectMarshaler, ObjectDemarshaler, TypeCode


@dataclass
class MarshallingInfo:
    __slots__ = ('marshaler', 'demarshaler', 'recursive_members')
    marshaler: ObjectMarshaler
    demarshaler: ObjectDemarshaler
    recursive_members: Optional[Set[AtomicData]]


class Marshal:
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


        Everything is either a:
            + passthrough type - untouched by marshaling, not referenced
            + reference atomic - like passthrough, but can be referenced (strings)
            + list - recursively handled
            + dict - recursively handled
            + object - converted to a dict with a type code


    """

    def __init__(
            self,
            type_field: str = '__type',
    ) -> None:
        self._type_field: str = type_field
        self._demarshaler_map: {TypeCode: MarshallingInfo} = {}
        self._marshaler_map: {Type: MarshallingInfo} = {}

        self.register_type(
            dict,
            Marshaler.marshal_mapping,

        )

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
        recursive_members = None if recursive_members is None else {m for m in recursive_members}
        marshalling_info = MarshallingInfo(marshaler, demarshaler, recursive_members)
        self._marshaler_map[target_type] = marshalling_info
        self._demarshaler_map[type_code] = marshalling_info

    def register_passthrough_type(self, target_type: Type) -> None:
        self.register_type(target_type, self._passthrough, self._passthrough)


    def marshal(self, source: any, use_references: bool = True, enable_root_references: bool = True) -> MarshaledData:
        marshaler_map = self._marshaler_map
        vertex_index = {} if use_references else None
        vertex_visited_set = set()

        referenced_types = set() # TODO

        def do_marshal(target: any, referencing: bool) -> any:
            target_type = type(target)
            target_id = id(target)
            marshalled = target

            # if there is a special marshaller, apply it
            marshalling_info = marshaler_map.get(target_type, None)
            if marshalling_info is not None:
                marshalled, is_passthrough, is_referencable = marshalling_info.marshaler(marshalled, do_marshal)

            if target is None or isinstance(target, bool) or isinstance(target, float):
                pass
            elif isinstance(target, int):
                if referencing:
                    raise ValueError('Integer encountered while marshalling a referencing target')
            else:
                if use_references:
                    if target_id not in vertex_index:
                        vertex_index[target_id] = len(vertex_index)  # index target
                    elif referencing:
                        return vertex_index[target_id]  # if target is already indexed, return its reference index

                if target_id in vertex_visited_set:  # detect cycles not elided by referencing
                    raise ValueError('Circular reference detected')

                if not isinstance(target, str):  # recur into Sequence and Mapping results
                    is_mapping = isinstance(target, Mapping)
                    is_iterable = not is_mapping and isinstance(target, Iterable)

                    vertex_visited_set.add(target_id)  # add to cycle detection set
                    if is_mapping:
                        recursive_members = None if not use_references or marshalling_info is None \
                            else marshalling_info.recursive_members
                        target = {k: do_marshal(v, recursive_members is not None and k in recursive_members)
                                  for k, v in sorted(target.items())}
                    elif isinstance(target, Iterable):
                        target = [do_marshal(e, referencing) for e in target]
                    else:
                        raise ValueError(f'Type has undefined marshalling protocol: "{target_type}"')

                    vertex_visited_set.remove(target_id)  # remove from cycle detection set
            return target

        return do_marshal(source, enable_root_references and use_references)

    def marshal(self, source: any, use_references: bool = True, enable_root_references: bool = True) -> MarshaledData:
        marshaler_map = self._marshaler_map
        vertex_index = {} if use_references else None
        vertex_visited_set = set()

        def do_marshal(target: any, referencing: bool) -> MarshaledData:
            target_type = type(target)

            # marshal target into a plain object
            marshalling_info = marshaler_map.get(target_type, None)
            if marshalling_info is not None:
                target = marshalling_info.marshaler(target)

            if target is None or isinstance(target, bool) or isinstance(target, float):
                pass
            elif isinstance(target, int):
                if referencing:
                    raise ValueError('Integer encountered while marshalling a referencing target')
            else:
                target_id = id(target)
                if use_references:
                    if target_id not in vertex_index:
                        vertex_index[target_id] = len(vertex_index)  # index target
                    elif referencing:
                        return vertex_index[target_id]  # if target is already indexed, return its reference index

                if target_id in vertex_visited_set:  # detect cycles not elided by referencing
                    raise ValueError('Circular reference detected')

                if not isinstance(target, str):  # recur into Sequence and Mapping results
                    is_sequence = isinstance(target, Sequence)
                    if not is_sequence and not isinstance(target, Mapping):
                        raise ValueError(f'Type has undefined marshalling protocol: "{target_type}"')

                    vertex_visited_set.add(target_id)  # add to cycle detection set
                    if is_sequence:
                        target = [do_marshal(e, referencing) for e in target]
                    else:
                        recursive_members = None if not use_references or marshalling_info is None \
                            else marshalling_info.recursive_members
                        target = {k: do_marshal(v, recursive_members is not None and k in recursive_members)
                                  for k, v in sorted(target.items())}
                    vertex_visited_set.remove(target_id)  # remove from cycle detection set
            return target

        return do_marshal(source, enable_root_references and use_references)

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
