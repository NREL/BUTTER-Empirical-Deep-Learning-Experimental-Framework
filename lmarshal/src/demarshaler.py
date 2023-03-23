from functools import singledispatchmethod
from typing import Dict, Mapping, Iterator, Union, Tuple, Type, Any

from .common_marshaler import CommonMarshaler
from .marshal_config import MarshalConfig
from .marshal_types import TypeCode, RawObjectDemarshaler, DemarshalingFactory, \
    DemarshalingInitializer


class Demarshaler(CommonMarshaler):
    __slots__ = ['_reference_index', '_type_map', '_result']

    def __init__(
        self,
        config: MarshalConfig,
        type_map: Dict[TypeCode, RawObjectDemarshaler],
        source: Any,
    ) -> None:
        super().__init__(config)
        self._type_map: Dict[TypeCode, RawObjectDemarshaler] = type_map
        self._reference_index: Dict[str, Any] = {}
        self._result: Any = self.demarshal(source)

    def __call__(self) -> Any:
        return self._result

    @singledispatchmethod
    def demarshal(self, source: Any) -> Any:
        raise ValueError(
            f'Type has undefined demarshaling protocol: "{type(source)}".')

    @demarshal.register(type(None))
    @demarshal.register(bool)
    @demarshal.register(int)
    @demarshal.register(float)
    def _(
        self,
        source: Union[None, bool, int, float],
    ) -> Union[None, bool, int, float]:
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
            self, source, lambda d, s: [], lambda d, s, r: r.extend(
                (d.demarshal(e) for e in s)))

    @demarshal.register
    def _(self, source: dict):
        # check for implicit references
        label_key = self._config.label_key
        if label_key in source:
            label = source[label_key]
            if label in self._reference_index:
                return self._reference_index[label]

        # demarshal typed dicts
        if self._config.type_key in source:
            type_code = source[self._config.type_key]
            if type_code not in self._type_map:
                raise ValueError(f'Unknown type code "{type_code}".')
            return self._type_map[type_code](self, source)

        # demarshal untyped dicts
        return Demarshaler.demarshal_dict(self, source)

    @staticmethod
    def demarshal_dict(demarshaler: 'Demarshaler', source: dict) -> dict:
        return Demarshaler.demarshal_typed(
            demarshaler, source, lambda d, s: {}, lambda d, s, r: r.update(
                demarshaler.dict_demarshaling_generator(s)))

    def dict_demarshaling_generator(
        self,
        source: Mapping,
    ) -> Iterator[Tuple[Any, Any]]:
        if self._config.flat_dict_key in source:
            # demarshal flattened key value pairs
            yield from ((self.demarshal(kvp[0]), self.demarshal(kvp[1]))
                        for kvp in source[self._config.flat_dict_key])

        # yield from ((k, self.demarshal(v)) for k, v in sorted((
        #     (self.demarshal_key(k), v) for k, v in source.items()
        #     if k not in self._config.control_key_set)))
        yield from ((self.demarshal_key(k), self.demarshal(v))
                    for k, v in sorted(source.items())
                    if k not in self._config.control_key_set)

    @staticmethod
    def demarshal_typed(
        demarshaler: 'Demarshaler',
        source: Any,
        factory: DemarshalingFactory,
        initializer: DemarshalingInitializer,
    ) -> Any:
        result = factory(demarshaler, source)
        demarshaler._register_label(result)
        initialized_dest = initializer(demarshaler, source, result)
        return result if initialized_dest is None else initialized_dest

    @staticmethod
    def default_object_factory(
        demarshaler: 'Demarshaler',
        source: Any,
        target_type: Type,
    ) -> Any:
        return target_type.__new__(target_type)

    @staticmethod
    def default_object_initializer(
        demarshaler: 'Demarshaler',
        source: Any,
        result: Any,
    ) -> None:
        for k, v in demarshaler.dict_demarshaling_generator(source):
            setattr(result, k, v)

    @staticmethod
    def custom_marshalable_initializer(
        demarshaler: 'Demarshaler',
        source: Any,
        result: Any,
    ) -> None:
        result.demarshal(dict(demarshaler.dict_demarshaling_generator(source)))

    @staticmethod
    def enum_factory(
        demarshaler: 'Demarshaler',
        source: Any,
        target_type: Type,
    ) -> Any:
        return target_type(
            demarshaler.demarshal(
                source[demarshaler.demarshal_key(demarshaler._config.enum_value_key)], ))

    @staticmethod
    def enum_initializer(
        demarshaler: 'Demarshaler',
        source: Any,
        result: Any,
    ) -> None:
        return result

    # @staticmethod
    # def default_dataclass_initializer(demarshaler: 'Demarshaler', source: Any, result: Any) -> None:
    #     #dataclasses.fields(C)
    #     kwargs = dict(demarshaler.dict_demarshaling_generator(source))
    #     result.__init__(**kwargs)

    @staticmethod
    def initialize_type_map(
        type_map: Dict[TypeCode, RawObjectDemarshaler],
        config: MarshalConfig,
    ) -> None:
        pass

    def _register_label(self, element: Any) -> Any:
        label = self._make_label(len(self._reference_index))
        self._reference_index[label] = element
        return element
