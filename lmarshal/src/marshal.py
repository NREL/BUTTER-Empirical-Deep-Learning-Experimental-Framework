from typing import Any, Dict, Iterable, Optional, Type

from lmarshal.src.custom_marshalable import CustomMarshalable

from .tuple_marshaling import demarshal_tuple, initialize_tuple, marshal_tuple
from .demarshaler import Demarshaler
from .marshal_config import MarshalConfig
from .marshaler import Marshaler
from .marshal_types import ObjectDemarshaler, ObjectMarshaler, TypeCode, DemarshalingFactory, DemarshalingInitializer


class Marshal:
    """

    """

    def __init__(
        self,
        config: Optional[MarshalConfig] = None,
    ) -> None:
        config = config if config is not None else MarshalConfig()

        self._config: MarshalConfig = config
        self._marshaler_type_map: Dict[Type, ObjectMarshaler] = {}
        self._demarshaler_type_map: Dict[TypeCode, ObjectDemarshaler] = {}

        Marshaler.initialize_type_map(self._marshaler_type_map, config)
        Demarshaler.initialize_type_map(self._demarshaler_type_map, config)

        # register tuple
        self.register_type(
            tuple,
            self._config.tuple_type_code,
            marshal_tuple,
            demarshal_tuple,
            initialize_tuple,
        )

        # register set
        self.register_type(
            set,
            self._config.set_type_code,
            lambda m, s:
            {config.flat_dict_key: Marshaler.marshal_list(m, (e for e in s))},
            lambda d, s: set(),
            lambda d, s, r: r.update(d.demarshal(s[config.flat_dict_key])),
        )

    def register_types(
        self,
        target_types: Iterable[Type],
    ) -> None:
        for t in target_types:
            self.register_type(t)

    def register_type(
        self,
        target_type: Type,
        type_code: Optional[TypeCode] = None,
        object_marshaler: Optional[ObjectMarshaler] = None,
        demarshaling_factory: Optional[DemarshalingFactory] = None,
        demarshaling_initializer: Optional[DemarshalingInitializer] = None,
    ) -> None:
        type_code = target_type.__name__ if type_code is None else type_code
        if object_marshaler is None:
            if issubclass(target_type, CustomMarshalable):
                object_marshaler = lambda m, s: s.marshal(m)
            else:
                object_marshaler = Marshaler.default_object_marshaler

        if demarshaling_factory is None:
            demarshaling_factory = \
                lambda demarshaler, source: Demarshaler.default_object_factory(
                demarshaler, source, target_type)

        if demarshaling_initializer is None:
            if issubclass(target_type, CustomMarshalable):
                demarshaling_initializer = lambda d, s, r: r.demarshal(d, s)
            # if dataclasses.is_dataclass(target_type):
            #     demarshaling_initializer = Demarshaler.default_dataclass_initializer
            else:
                demarshaling_initializer = Demarshaler.default_object_initializer

        if type_code is None:
            raise NotImplementedError()  # should never happen

        self._marshaler_type_map[target_type] = \
            lambda marshaler, source: Marshaler.marshal_typed(
                marshaler, source, type_code, object_marshaler)

        self._demarshaler_type_map[type_code] = \
            lambda demarshaler, source: Demarshaler.demarshal_typed(
            demarshaler, source, demarshaling_factory, demarshaling_initializer)

    def marshal(self, source: Any) -> Any:
        return Marshaler(self._config, self._marshaler_type_map, source)()

    def demarshal(self, source: Any) -> Any:
        return Demarshaler(self._config, self._demarshaler_type_map, source)()
