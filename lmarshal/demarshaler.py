from functools import singledispatchmethod
from typing import Mapping, Iterator, Union, Tuple, Type

from lmarshal.common_marshaler import CommonMarshaler
from lmarshal.marshal_config import MarshalConfig
from lmarshal.types import TypeCode, ObjectDemarshaler, DemarshalingFactory, \
    DemarshalingInitializer


class Demarshaler(CommonMarshaler):
    __slots__ = ['_reference_index', '_type_map', '_result']

    def __init__(self,
                 config: MarshalConfig,
                 type_map: {TypeCode: ObjectDemarshaler},
                 source: any,
                 ) -> None:
        super().__init__(config)
        self._type_map: {TypeCode: ObjectDemarshaler} = type_map
        self._reference_index: {str: any} = {}
        self._result: any = self.demarshal(source)

    def __call__(self) -> any:
        return self._result

    @singledispatchmethod
    def demarshal(self, source: any) -> any:
        raise ValueError(f'Type has undefined demarshaling protocol: "{type(source)}".')

    @demarshal.register(type(None))
    @demarshal.register(bool)
    @demarshal.register(int)
    @demarshal.register(float)
    def _(self, source: Union[None, bool, int, float]) -> Union[None, bool, int, float]:
        return source

    @demarshal.register
    def _(self, source: str):
        if source.startswith(self._config.reference_prefix):
            label = source[len(self._config.reference_prefix):]
            if label not in self._reference_index:
                raise ValueError(f'Encountered undefined reference "{label}"')
            return self._reference_index[label]  # dereference: return referent
        if source.startswith(self._config.escape_prefix):
            source = self._unescape_string(source)
        if self._config.reference_strings:
            self._register_label(source)
        return source

    @demarshal.register
    def _(self, source: list):
        return Demarshaler.demarshal_typed(
            self,
            source,
            lambda d, s: [],
            lambda d, s, r: r.extend((d.demarshal(e) for e in s))
        )

    @demarshal.register
    def _(self, source: dict):
        # demarshal typed dicts
        if self._config.type_key in source:
            type_code = source[self._config.type_key]
            if type_code not in self._type_map:
                raise ValueError(f'Unknown type code "{type_code}".')
            return self._type_map[type_code](self, source)

        # demarshal untyped dicts
        return Demarshaler.demarshal_typed(
            self,
            source,
            lambda d, s: {},
            lambda d, s, r: r.update(self.dict_demarshaling_generator(s)))

    def dict_demarshaling_generator(self, source: Mapping) -> Iterator[Tuple[any, any]]:
        if self._config.flat_dict_key in source:  # demarshal flattened key value pairs
            items = self.demarshal(source[self._config.flat_dict_key])
            if not isinstance(items, list):
                raise ValueError(
                    f'Found a {type(items)} instead of a list while demarshaling a flattened dict.')
            for item in items:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError('Expected a list of length 2, but found something else.')
                yield tuple(item)

        yield from ((self.demarshal_key(k), self.demarshal(v))
                    for k, v in sorted(source.items())
                    if k not in self._config.control_key_set)

    def demarshal_key(self, source: str) -> str:
        return self._unescape_string(source) if source.startswith(self._config.escape_prefix) else source

    @staticmethod
    def demarshal_typed(
            demarshaler: 'Demarshaler',
            source: any,
            factory: DemarshalingFactory,
            initializer: DemarshalingInitializer,
    ) -> any:
        result = factory(demarshaler, source)
        demarshaler._register_label(result)
        initialized_dest = initializer(demarshaler, source, result)
        return result if initialized_dest is None else initialized_dest

    @staticmethod
    def default_object_factory(demarshaler: 'Demarshaler', source: any, target_type: Type) -> {}:
        return target_type.__new__(target_type)

    @staticmethod
    def default_object_initializer(demarshaler: 'Demarshaler', source: any, result: any) -> None:
        for k, v in demarshaler.dict_demarshaling_generator(source):
            setattr(result, k, v)

    @staticmethod
    def initialize_type_map(type_map: {TypeCode: ObjectDemarshaler}, config: MarshalConfig) -> None:
        pass

    def _register_label(self, element: any) -> any:
        label = self._make_label(len(self._reference_index))
        self._reference_index[label] = element
        return element
