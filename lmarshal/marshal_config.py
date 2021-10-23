from dataclasses import field, dataclass
from typing import Type, Optional, Iterable, Mapping, Callable, MutableMapping

from lmarshal.types import ObjectMarshaler, TypeCode, ObjectDemarshaler


@dataclass
class MarshalConfig:
    marshaler_type_map: {Type: ObjectMarshaler} = field(default_factory=dict)
    demarshaler_map: {Type: ObjectDemarshaler} = field(default_factory=dict)
    demarshaler_type_code_map: {TypeCode: ObjectDemarshaler} = field(default_factory=dict)

    type_key: str = '%'
    label_key: str = '&'
    reference_prefix: str = '*'
    escape_prefix: str = '!'
    flat_dict_key: str = ':'
    reserved_prefixes_and_keys: [str] = \
        field(default_factory=lambda: ['@', '#', '$', '%', '^', '&', '*', '=', '|'])
    # implicit_reference_char_map :[str] = \
    #     field(default_factory=lambda: list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

    control_key_set: {str} = field(init=False)
    control_prefix_set: {str} = field(init=False)

    def __post_init__(self):
        if len(self.reference_prefix) <= 0 or len(self.escape_prefix) <= 0:
            raise ValueError('Prefixes are zero length.')

        self.control_key_set = {self.type_key, self.label_key, self.flat_dict_key}
        if len(self.control_key_set) != 3 or self.escape_prefix in self.control_key_set:
            raise ValueError('Control and escape keys are not distinct.')
        self.control_key_set.update(self.reserved_prefixes_and_keys)

        self.control_prefix_set = {self.reference_prefix, self.escape_prefix}
        if len(self.control_prefix_set) != 2:
            raise ValueError('Control prefixes are not distinct.')
        self.control_prefix_set.update(self.reserved_prefixes_and_keys)

        # self.register_type(NoneType, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        # self.register_type(bool, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        # self.register_type(int, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        # self.register_type(float, Marshaler.marshal_passthrough, Demarshaler.demarshal_passthrough)
        # TODO

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
        self.marshaler_type_map[target_type] = marshaler
        self.demarshaler_type_map[type_code] = demarshaler
        # TODO: type code and type handler for demarshaling....

    def make_implicit_label(self, element_number: int) -> str:
        return hex(element_number)[2:]

    def make_default_marshaler(self, target_type: Type) -> ObjectMarshaler:
        pass

    def make_default_demarshaler(self, target_type: Type) -> ObjectDemarshaler:
        pass
    
    
