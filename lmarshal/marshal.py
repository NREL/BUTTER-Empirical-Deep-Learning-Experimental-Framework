from types import Union, NoneType
from typing import Optional, Type, Mapping, Iterable

from lmarshal.demarshaler import Demarshaler
from lmarshal.marshaler import Marshaler
from lmarshal.types import ObjectMarshaler, ObjectDemarshaler, TypeCode


class Marshal:
    """

    """

    def __init__(
            self,
            type_key: str = '%',
            label_key: str = '&',
            reference_prefix: str = '*',
            escape_prefix: str = '!',
            flat_dict_key: str = ':',
    ) -> None:
        if len(reference_prefix) != 1 or len(escape_prefix) != 1:
            raise ValueError('Prefixes are not length 1')

        self._marshaler_map: {Type: ObjectMarshaler} = {}
        self._demarshaler_map: {TypeCode: ObjectDemarshaler} = {}

        self._type_key: str = type_key
        self._label_key: str = label_key
        self._reference_prefix: str = reference_prefix
        self._escape_prefix: str = escape_prefix
        self._flat_dict_key: str = flat_dict_key

        self._prefix_key_set: {str} = {type_key, label_key, flat_dict_key, escape_prefix}
        self._prefix_value_set: {str} = {reference_prefix, escape_prefix}
        if len(self._prefix_key_set) != 3 or len(self._prefix_value_set) != 2:
            raise ValueError('Prefixes are not distinct')

        self.register_type(NoneType, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        self.register_type(bool, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        self.register_type(int, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        self.register_type(float, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)

        def demarshal_list(demarshaler: Demarshaler, target: Iterable, referencing: bool) -> list:
            demarshaled = []
            demarshaler._vertex_index.append(demarshaled)
            for e in target:
                demarshaled.append(demarshaler.demarshal(e, referencing))
            return demarshaled

        self.register_type(list, marshal_list, demarshal_list)

        def marshal_string(
                marshaler: marshaler,
                target: str,
                referencing: bool,
        ) -> Union[str, int]:
            if target.startswith(
                    marshaler._marshal._reference_prefix):  # escape strings that would otherwise be conflated with references
                target = ref_escape + target
            return marshaler.marshal_object(target, referencing, lambda marshalled: marshalled)

        def demarshal_string(demarshaler: 'Demarshaler', target: str, referencing: bool) -> str:
            ref_prefix = '@'
            ref_escape = '@@'
            if target.startswith(ref_prefix):
                if target.startswith(ref_escape):  # un-escape escaped strings
                    target = target[len(ref_escape):]
                else:
                    index = target[len(ref_prefix):]
                    target = demarshaler._vertex_index[index]
            demarshaler._vertex_index.append(target)
            return target

        def marshal_dict(marshaler: marshaler, target: Mapping, referencing: bool) -> Union[dict, int]:
            return marshaler.marshal_object(
                target,
                referencing,
                lambda marshalled: {
                    k: marshaler.marshal(v, referencing)
                    for k, v in sorted(marshalled.items())})

        def demarshal_dict(demarshaler: 'Demarshaler', target: Mapping, referencing: bool) -> any:
            type_code = target.get(demarshaler._type_field, None)
            if type_code not in demarshaler._demarshaler_object_handlers:
                raise ValueError(f'Unknown type code: "{type_code}"')
            handler = demarshaler._demarshaler_object_handlers[type_code]

            accumulator, demarshaled = handler.make_empty()
            demarshaler._vertex_index.append(demarshaled)
            for k, v in sorted(target.items()):
                # k = demarshaler.demarshal(k, False)
                accumulator[k] = demarshaler.demarshal(v, referencing)
            demarshaled = handler.demarshal(accumulator, demarshaled)
            return demarshaled

        def marshal_dataclass():
            pass

    def marshal(self, target: any) -> any:
        # marshal target into a plain object
        target_type = type(target)
        if target_type not in self._marshaler_map:
            raise ValueError(f'Type has undefined marshaling protocol: "{target_type}"')
        return self._marshaler_map[target_type](self, target)

    def marshal_object(self, target: any, handler) -> any:
        target_id = id(target)

        if target_id not in self._vertex_index:
            self._vertex_index[target_id] = len(self._vertex_index)  # index target
        else:
            # if target is already indexed, return its reference index
            index = self._vertex_index[target_id]
            return

        self._vertex_visited_set.add(target_id)  # add to cycle detection set
        marshaled = handler(self, target, referencing)  # marshal target ...
        self._vertex_visited_set.remove(target_id)  # remove from cycle detection set
        return marshaled

    def register_type(
            self,
            target_type: Type,
            marshaler: Optional[ObjectMarshaler] = None,
            demarshaler: Optional[ObjectDemarshaler] = None,
            type_code: Optional[TypeCode] = None,
    ) -> None:
        marshaler = self.make_default_marshaler(target_type) if marshaler is None else marshaler
        demarshaler = self.make_default_demarshaler(target_type) if demarshaler is None else demarshaler
        type_code = target_type.__name__ if type_code is None else type_code
        self._marshaler_map[target_type] = marshaler
        self._demarshaler_map[type_code] = demarshaler

    def make_default_marshaler(self, target_type: Type) -> ObjectMarshaler:
        pass

    def make_default_demarshaler(self, target_type: Type) -> ObjectDemarshaler:
        pass
