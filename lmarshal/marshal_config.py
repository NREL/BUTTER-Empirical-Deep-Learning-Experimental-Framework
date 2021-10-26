from types import NoneType
from typing import Type, Mapping, Iterable, Iterator, MutableMapping, Optional

from lmarshal.types import ObjectMarshaler, TypeCode, ObjectDemarshaler, DemarshalingFactory, \
    DemarshalingInitializer


# type_key: str = '%'
# label_key: str = '&'
# reference_prefix: str = '*'
# escape_prefix: str = '!'
# flat_dict_key: str = ':'
# reserved_prefixes_and_keys = ['@', '#', '$', '%', '^', '&', '*', '=', '|']
# control_key_set = {type_key, label_key, flat_dict_key}
# control_key_set.update(reserved_prefixes_and_keys)
# control_prefix_set = {reference_prefix, escape_prefix}
# control_prefix_set.update(reserved_prefixes_and_keys)
# 

# @dataclass
# noinspection PyProtectedMember
class MarshalConfig:
    # marshaler_type_map: {Type: ObjectMarshaler} = field(default_factory=dict)
    # demarshaler_map: {Type: ObjectDemarshaler} = field(default_factory=dict)
    # demarshaler_type_code_map: {TypeCode: ObjectDemarshaler} = field(default_factory=dict)
    #
    # type_key: str = '%'
    # label_key: str = '&'
    # reference_prefix: str = '*'
    # escape_prefix: str = '!'
    # flat_dict_key: str = ':'
    # reserved_prefixes_and_keys: [str] = \
    #     field(default_factory=lambda: ['!', '@', '#', '$', '%', '^', '&', '*', '=', '|', '<', '>', '?', ';', ':'])
    # # implicit_reference_char_map :[str] = \
    # #     field(default_factory=lambda: list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    #
    # control_key_set: {str} = field(init=False)
    # control_prefix_set: {str} = field(init=False)
    #
    # def __post_init__(self):
    #     if len(self.reference_prefix) <= 0 or len(self.escape_prefix) <= 0:
    #         raise ValueError('Prefixes are zero length.')
    #
    #     self.control_key_set = {self.type_key, self.label_key, self.flat_dict_key}
    #     if len(self.control_key_set) != 3 or self.escape_prefix in self.control_key_set:
    #         raise ValueError('Control and escape keys are not distinct.')
    #     self.control_key_set.update(self.reserved_prefixes_and_keys)
    #
    #     self.control_prefix_set = {self.reference_prefix, self.escape_prefix}
    #     if len(self.control_prefix_set) != 2:
    #         raise ValueError('Control prefixes are not distinct.')
    #     self.control_prefix_set.update(self.reserved_prefixes_and_keys)

    def __init__(self,
                 type_key: str = '%',
                 label_key: str = '&',
                 reference_prefix: str = '*',
                 escape_prefix: str = '!',
                 flat_dict_key: str = ':',
                 reserved_prefixes_and_keys: Iterable[str] =
                 ('!', '@', '#', '$', '%', '^', '&', '*', '=', '|', '<', '>', '?', ';', ':'),
                 label_dicts: bool = False,
                 label_referenced: bool = True,
                 always_use_references: bool = True,
                 reference_strings: bool = False,
                 reference_lists: bool = True,
                 reference_dicts: bool = True,
                 ):

        self._label_key: str = label_key
        self._label_dicts: bool = label_dicts
        self._label_referenced: bool = label_referenced

        if len(reference_prefix) <= 0 or len(escape_prefix) <= 0:
            raise ValueError('Prefixes are zero length.')

        control_key_set = {type_key, label_key, flat_dict_key}
        if len(control_key_set) != 3 or escape_prefix in control_key_set:
            raise ValueError('Control and escape keys are not distinct.')
        control_key_set.update(reserved_prefixes_and_keys)

        control_prefix_set = {reference_prefix, escape_prefix}
        if len(control_prefix_set) != 2:
            raise ValueError('Control prefixes are not distinct.')
        control_prefix_set.update(reserved_prefixes_and_keys)

        self._marshaler_type_map: {Type: ObjectMarshaler} = {}
        self._demarshaler_map: {Type: ObjectDemarshaler} = {}
        self._demarshaler_type_code_map: {TypeCode: ObjectDemarshaler} = {}

        # ---------- common functions

        def passthrough(marshaler_or_demarshaler, source: any) -> any:
            return source

        def make_implicit_label(index: int):
            return hex(index)[2:]

        def escape_string(source: str) -> str:
            return escape_prefix + source

        def unescape_string(source: str) -> str:
            return source[1:]

        # ---------- marshaling functions

        class Marshaler:
            def __init__(self, marshaler_type_map: {Type: ObjectMarshaler}, source: any) -> None:
                self._marshaler_type_map: {Type: ObjectMarshaler} = marshaler_type_map
                self._vertex_index: {int: (str, any)} = {}
                self._referenced: {int} = set()
                self._result: any = self.marshal(source)
                if label_referenced and not label_dicts:
                    for element_id in self._referenced:
                        label, element = self._vertex_index[element_id]
                        if isinstance(element, dict):
                            element[label_key] = label

            def __call__(self):
                return self._result

            def marshal(self, source: any) -> any:
                # marshal source into a plain object
                source_type = type(source)
                marshaler_type_map = self._marshaler_type_map
                if source_type not in self._marshaler_type_map:
                    raise ValueError(f'Type has undefined marshaling protocol: "{source_type}".')
                return marshaler_type_map[source_type](self, source)

            def marshal_referencable(self, source: any, marshaler: ObjectMarshaler) -> any:
                vertex_index = self._vertex_index
                source_id = id(source)
                if source_id in vertex_index:  # if source is already indexed, return its reference index
                    label, source = vertex_index[source_id]
                    if always_use_references or source is None:
                        self._referenced.add(source_id)
                        return reference_prefix + label
                label = make_implicit_label(len(vertex_index))
                vertex_index[source_id] = (label, None)
                dest = marshaler(self, source)
                if label_dicts and isinstance(dest, dict):
                    dest[label_key] = label
                vertex_index[source_id] = (label, dest)
                return dest

            def marshal_bare_string(self, source):
                return escape_string(source) if len(source) > 0 and source[0] in control_prefix_set else source

            def marshal_bare_list(self, source):
                return [self.marshal(e) for e in source]

            def marshal_bare_dict(self, source: Mapping) -> any:
                items = []
                for k, v in source.items():
                    if not isinstance(k, str):
                        return {flat_dict_key: self.marshal([[k, v] for k, v in source.items()])}
                    if k in control_key_set or k.startswith(escape_prefix):
                        k = escape_string(k)  # TODO: optionally allow referenced strings as keys?
                    items.append((k, v))
                return {k: self.marshal(v) for k, v in sorted(items)}

            def marshal_object(self, source: any, type_code: TypeCode, marshaler: ObjectMarshaler) -> any:
                marshaled = marshaler(self, source)
                if not isinstance(marshaled, dict):
                    raise ValueError(
                        f'ObjectMarshaler for type {type(source)} returned a {type(marshaled)} instead of a dict.')
                marshaled[type_key] = type_code
                return marshaled

        self._marshaller = Marshaler

        # ---------------- demarshaling functions
        class Demarshaler:
            def __init__(self,
                         demarshaler_map: {Type: ObjectDemarshaler},
                         type_demarshaler_map: {TypeCode: ObjectDemarshaler},
                         source: any) -> None:
                self._reference_index: {str: any} = {}
                self._demarshaler_map: {Type: ObjectDemarshaler} = demarshaler_map
                self._type_demarshaler_map: {TypeCode: ObjectDemarshaler} = type_demarshaler_map
                self._result: any = self.demarshal(source)

            def __call__(self) -> any:
                return self._result

            def demarshal(self, source: any) -> any:
                source_type = type(source)
                if source_type not in self._demarshaler_map:
                    raise ValueError(f'Type has undefined demarshaling protocol: "{source_type}".')
                return self._demarshaler_map[source_type](self, source)

            def demarshal_bare_string(self, source: str) -> str:
                if source.startswith(reference_prefix):
                    label = source[len(reference_prefix):]
                    if label not in self._reference_index:
                        raise ValueError(f'Encountered undefined reference "{label}"')
                    return self._reference_index[label]  # dereference: return referent
                if source.startswith(escape_prefix):
                    source = unescape_string(source)
                # self.register_label(source)
                return source

            def initialize_bare_list(self, source: Iterable, dest: [any]) -> None:
                dest.extend((self.demarshal(e) for e in source))

            def initialize_bare_dict(self, source: Mapping, dest: MutableMapping) -> None:
                dest.update(self.dict_demarshaling_generator(source))

            def initialize_regular_object(self, source: Mapping, dest: any) -> None:
                for member_name, member_value in self.dict_demarshaling_generator(source):
                    setattr(dest, member_name, member_value)

            def demarshal_dict(self, source: Mapping) -> any:
                if type_key in source:  # demarshal typed dicts
                    type_code = source[type_key]
                    if type_code not in self._type_demarshaler_map:
                        raise ValueError(f'Unknown type code "{type_code}".')
                    return self._type_demarshaler_map[type_code](source)
                return self.demarshal_composite(  # demarshal untyped dicts
                    source,
                    lambda demarshaler, source: {},
                    Demarshaler.initialize_bare_dict)

            def register_label(self, element: any) -> any:
                label = make_implicit_label(len(self._reference_index))
                self._reference_index[label] = element
                return element

            def demarshal_composite(self,
                                    source: Mapping,
                                    factory: DemarshalingFactory,
                                    initializer: DemarshalingInitializer) \
                    -> any:
                dest = factory(self, source)
                self.register_label(dest)
                initializer(self, source, dest)
                return dest

            def demarshal_key(self, source: str) -> str:
                return unescape_string(source) if source.startswith(escape_prefix) else source

            def dict_demarshaling_generator(self, source: Mapping) -> Iterator[(any, any)]:
                if flat_dict_key in source:  # demarshal flattened key value pairs
                    items = source[flat_dict_key]
                    if not isinstance(items, list):
                        raise ValueError(
                            f'Found a {type(items)} instead of a list while demarshaling a flattened dict.')
                    for item in items:
                        if not isinstance(item, list) or len(item) != 2:
                            raise ValueError('Expected a list of length 2, but found something else.')
                        yield self.demarshal(item[0]), self.demarshal(item[1])

                yield from ((self.demarshal_key(k), self.demarshal(v))
                            for k, v in sorted(source.items())
                            if k not in control_key_set)

        self._demarshaler = Demarshaler

        # register basic types
        self.register_primitive_type(NoneType, passthrough, passthrough)
        self.register_primitive_type(bool, passthrough, passthrough)
        self.register_primitive_type(int, passthrough, passthrough)
        self.register_primitive_type(float, passthrough, passthrough)

        if reference_strings:
            self.register_primitive_type(
                str,
                lambda marshaler, source: marshaler.marshal_referencable(source, Marshaler.marshal_bare_string),
                lambda demarshaler, source: demarshaler.register_label(demarshaler.demarshal_bare_string(source)))
        else:
            self.register_primitive_type(str, Marshaler.marshal_bare_string, Demarshaler.demarshal_bare_string)

        self.register_primitive_type(
            list,
            lambda marshaler, source: marshaler.marshal_referencable(source, Marshaler.marshal_bare_list),
            lambda demarshaler, source: demarshaler.demarshal_composite(
                source,
                lambda demarshaler, source: [],
                Demarshaler.initialize_bare_list))

        self.register_primitive_type(
            dict,
            lambda marshaler, source: marshaler.marshal_referencable(source, Marshaler.marshal_bare_dict),
            Demarshaler.demarshal_dict)

    def register_primitive_type(
            self,
            target_type: Type,
            marshaler: ObjectMarshaler = None,
            demarshaler: ObjectDemarshaler = None,
    ) -> None:
        self._marshaler_type_map[target_type] = marshaler
        self._demarshaler_map[target_type] = demarshaler

    def register_coded_type(
            self,
            target_type: Type,
            object_marshaler: Optional[ObjectMarshaler] = None,
            demarshaling_factory: Optional[DemarshalingFactory] = None,
            demarshaling_initializer: Optional[DemarshalingInitializer] = None,
            type_code: Optional[TypeCode] = None,
    ) -> None:
        self._marshaler_type_map[target_type] = lambda marshaler, source: \
            marshaler.marshal_object(source, type_code, object_marshaler)
        type_code = target_type.__name__ if type_code is None else type_code
        self._demarshaler_type_code_map[type_code] = lambda demarshaler, source: \
            demarshaler.demarshal_composite(source, demarshaling_factory, demarshaling_initializer)

    def marshal(self, source: any) -> any:
        return self._marshaller(self._marshaler_type_map, source)()

    def demarshal(self, source: any) -> any:
        return self._demarshaler(self._demarshaler_map, self._demarshaler_type_code_map, source)()
