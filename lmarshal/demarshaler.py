from typing import Iterable, Mapping, Type, Iterator

from lmarshal.marshal_config import MarshalConfig
from lmarshal.types import ObjectDemarshalingFactory, \
    ObjectDemarshalingInitializer


class Demarshaler:
    """
    + construct empty elements (empty constructor)
    + set elements / references (setter function)
    + call post demarshal initializers (post demarshal hook)


    """

    def __init__(self, config: MarshalConfig, source: any) -> None:
        self._config: MarshalConfig = config
        self._reference_index: {str: any} = {}
        self._result: any = self.demarshal(source)

    def __call__(self) -> any:
        return self._result

    def demarshal(self, source: any) -> any:
        source_type = type(source)
        demarshaler_map = self._config.demarshaler_map
        if source_type not in demarshaler_map:
            raise ValueError(f'Type has undefined demarshaling protocol: "{source_type}".')
        return demarshaler_map[source_type](self, source)

    def demarshal_passthrough(self, source: any) -> any:
        return source

    def demarshal_string(self, source: str) -> str:
        reference_prefix = self._config.reference_prefix
        if source.startswith(reference_prefix):
            label = source[len(reference_prefix):]
            if label in self._reference_index:
                return self._reference_index[label]  # dereference: return referent
            raise ValueError(f'Encountered undefined reference "{label}"')
        if source.startswith(self._config.escape_prefix):
            source = self.unescape_string(source)
        return self.register_implicit_label(source)

    def unescape_string(self, source: str) -> str:
        return source[1:]

    def register_implicit_label(self, element: any) -> any:
        return self.register_label(self._config.make_implicit_label(len(self._reference_index)), element)

    def make_implicit_label(self, source: any) -> str:
        return self._config.make_implicit_label(len(self._reference_index))

    def register_label(self, element: any, label: str) -> any:
        self._reference_index[label] = element
        return element

    def demarshal_composite(self,
                            source: any,
                            make: ObjectDemarshalingFactory,
                            get_label,
                            initialize: ObjectDemarshalingInitializer):
        result = make(self, source)
        self.register_label(result, get_label(self, source))
        initialize(self, source, result)
        return result

    def demarshal_list(self, source: Iterable) -> []:
        return self.demarshal_composite(
            source,
            lambda marshaller, source: [],
            self.make_implicit_label,
            self.list_initializer,
        )

    def list_initializer(self, source: Iterable, result: []) -> None:
        result.extend((self.demarshal(e) for e in source))

    def demarshal_dict(self, source: Mapping) -> any:
        type_key = self._config.type_key
        if type_key not in source:
            return self.demarshal_plain_dict(source)  # demarshal untyped dicts

        if self._config.label_key in source:
            label = source[self._config.label_key]

        # demarshal typed dicts
        type_code = source[type_key]
        type_code_map = self._config.demarshaler_type_code_map
        if type_code in type_code_map:
            return type_code_map[type_code](source)
        raise ValueError(f'Unknown type code "{type_code}".')

    def demarshal_plain_dict(self, source: Mapping) -> any:
        return self.demarshal_composite(source, lambda marshaller, source: {}, self.dict_initializer)

    def dict_initializer(self, source: Mapping, result: {}) -> None:
        result.update(self.dict_demarshaling_generator(source))

    def dict_demarshaling_generator(self, source: Mapping) -> Iterator[(any, any)]:
        flat_dict_key = self._config.flat_dict_key
        if flat_dict_key in source:  # demarshal flattened key value pairs
            items = source[flat_dict_key]
            if not isinstance(items, list):
                raise ValueError(f'Found a {type(items)} instead of a list while demarshaling a flattened dict.')
            for item in items:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError('Expected a list of length 2, but found something else.')
                yield self.demarshal(item[0]), self.demarshal(item[1])

        # TODO: optionally allow referenced strings as keys?
        control_key_set = self._config.control_key_set
        yield from ((self.demarshal_key(k), self.demarshal(v))
                    for k, v in sorted(source.items())
                    if k not in control_key_set)

    def demarshal_key(self, source: str) -> str:
        if source.startswith(self._config.escape_prefix):
            return self.unescape_string(source)
        return source

    def make_regular_object_demarshaler(self, type: Type) -> any:
        return self.demarshal_composite(
            source,
            lambda demarshaller, source: type.__new__(),
            self.initialize_regular_object
        )

    def initialize_regular_object(self, source: Mapping, result: any) -> None:
        for member_name, member_value in self.dict_demarshaling_generator(source):
            setattr(result, member_name, member_value)

    # # def demarshal_dict(self, source: Mapping) -> any:
    # #     type_key = self._config.type_key
    # #     if type_key in source:
    # #         # demarshal typed dicts
    # #         type_code = source[type_key]
    # #         type_code_map = self._config.demarshaler_type_code_map
    # #         if type_code not in type_code_map:
    # #             raise ValueError(f'Unknown type code "{type_code}".')
    # #         return type_code_map[type_code](source)
    # #
    # #     # demarshal untyped dicts
    # #     result = self.register_implicit_label({})
    # #     label_key = self._config.label_key
    # #     if label_key in source:  # register explicit label
    # #         self.register_label(result, source[label_key])
    # #
    # #     result.update(self.dict_demarshaling_generator(source))
    # #     return result
    #
    # def demarshal_dict(self, source: Mapping) -> any:
    #     type_key = self._config.type_key
    #     if type_key in source:
    #         # demarshal typed dicts
    #         type_code = source[type_key]
    #         type_code_map = self._config.demarshaler_type_code_map
    #         if type_code not in type_code_map:
    #             raise ValueError(f'Unknown type code "{type_code}".')
    #         return type_code_map[type_code](source)
    #
    #     # demarshal untyped dicts
    #     result = self.register_implicit_label({})
    #     label_key = self._config.label_key
    #     if label_key in source:  # register explicit label
    #         self.register_label(result, source[label_key])
    #
    #     result.update(self.dict_demarshaling_generator(source))
    #     return result
    #

    #

    #
    # def register_implicit_label(self, element: any) -> any:
    #     return self.register_label(self._config.make_implicit_label(len(self._reference_index)), element)
    #
    # def register_label(self, element: any, label: str) -> any:
    #     self._reference_index[label] = element
    #     return element
    #
    # def demarshal_referencable(self, source: any, registrar: ReferenceRegistrar) -> any:
    #     if isinstance(source, str) and source.startswith(self._config.reference_prefix):  # if source is a reference:
    #         label = self.demarshal_reference(source)
    #         if label in self._reference_index:
    #             return self._reference_index[label]  # dereference: return referent
    #         raise ValueError(f'Encountered undefined reference "{label}"')
    #     # if source is not a reference, demarshal it
    #     return self.demarshal(source, self.register_implicit_label)
    #
    # def demarshal_reference(self, source: str) -> any:
    #     return source[len(self._config.reference_prefix):]
