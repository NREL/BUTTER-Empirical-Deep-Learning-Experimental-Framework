import ctypes
import dataclasses
from typing import Any, Dict, Optional, Type

from .demarshaler import Demarshaler
from .marshal_config import MarshalConfig
from .marshaler import Marshaler
from .types import ObjectDemarshaler, ObjectMarshaler, TypeCode, DemarshalingFactory, DemarshalingInitializer


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

        self.register_type(
            tuple,
            self._config.tuple_type_code,
            lambda m, s: {
                config.flat_dict_key: Marshaler.marshal_list(m, (e for e in s))},
            lambda d, s: tuple(s[config.flat_dict_key]),
            Marshal.initialize_tuple)

        self.register_type(
            set,
            self._config.set_type_code,
            lambda m, s: {
                config.flat_dict_key: Marshaler.marshal_list(m, (e for e in s))},
            lambda d, s: set(),
            lambda d, s, r: r.update(d.demarshal(s[config.flat_dict_key])))

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
            object_marshaler = Marshaler.default_object_marshaler

        if demarshaling_factory is None:
            def _demarshaling_factory(
                d, s): return Demarshaler.default_object_factory(d, s, target_type)
            demarshaling_factory = _demarshaling_factory

        if demarshaling_initializer is None:
            if dataclasses.is_dataclass(target_type):
                demarshaling_initializer = Demarshaler.default_dataclass_initializer
            else:
                demarshaling_initializer = Demarshaler.default_object_initializer

        if type_code is None:
            raise NotImplementedError() # should never happen

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

    @staticmethod
    def initialize_tuple(demarshaler: Demarshaler, source: dict, result: tuple) -> None:

        class APIFunctions:
            PyGILState_Ensure = staticmethod(
                ctypes.pythonapi.PyGILState_Ensure)
            PyGILState_Release = staticmethod(
                ctypes.pythonapi.PyGILState_Release)
            Py_DecRef = staticmethod(ctypes.pythonapi.Py_DecRef)
            Py_IncRef = staticmethod(ctypes.pythonapi.Py_IncRef)
            PyTuple_SetItem = staticmethod(ctypes.pythonapi.PyTuple_SetItem)

        values = demarshaler.demarshal(
            source[demarshaler._config.flat_dict_key])
        if not isinstance(values, list):
            raise TypeError(
                f'Found a {type(values)} when expecting list of values while demarshaling a tuple.')

        ref_count_ptr = ctypes.POINTER(ctypes.c_ssize_t)
        APIFunctions.PyGILState_Ensure()
        try:
            py_object_target = ctypes.py_object(result)
            ref_count = ctypes.cast(
                id(result), ref_count_ptr).contents.value - 1
            for _ in range(ref_count):
                APIFunctions.Py_DecRef(py_object_target)
            try:
                for i, v in enumerate(values):
                    value_py_object = ctypes.py_object(v)
                    APIFunctions.Py_IncRef(value_py_object)
                    if APIFunctions.PyTuple_SetItem(py_object_target, ctypes.c_ssize_t(i), value_py_object):
                        raise SystemError('Tuple mutation failed.')
            finally:
                for _ in range(ref_count):
                    APIFunctions.Py_IncRef(py_object_target)
        finally:
            APIFunctions.PyGILState_Release()
