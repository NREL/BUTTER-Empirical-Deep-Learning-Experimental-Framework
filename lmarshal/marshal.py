import ctypes
from typing import Optional, Type

from lmarshal.demarshaler import Demarshaler
from lmarshal.marshal_config import MarshalConfig
from lmarshal.marshaler import Marshaler
from lmarshal.types import ObjectDemarshaler, ObjectMarshaler, TypeCode, DemarshalingFactory, DemarshalingInitializer


class Marshal:
    """

    """

    def __init__(
            self,
            config: MarshalConfig,
    ) -> None:
        self._config: MarshalConfig = config
        self._marshaler_type_map: {Type: ObjectMarshaler} = {}
        self._demarshaler_type_map: {TypeCode: ObjectDemarshaler} = {}

        Marshaler.initialize_type_map(self._marshaler_type_map)
        Demarshaler.initialize_type_map(self._demarshaler_type_map)

        self.register_type(
            tuple,
            lambda m, s: {'': Marshaler.marshal_list(m, (e for e in s))},
            lambda d, s: tuple(s['']),
            Marshal._initialize_tuple,
            self._config.tuple_type_code)

    def register_type(
            self,
            target_type: Type,
            object_marshaler: Optional[ObjectMarshaler] = None,
            demarshaling_factory: Optional[DemarshalingFactory] = None,
            demarshaling_initializer: Optional[DemarshalingInitializer] = None,
            type_code: Optional[TypeCode] = None,
    ) -> None:
        type_code = target_type.__name__ if type_code is None else type_code
        self._marshaler_type_map[target_type] = \
            lambda marshaler, source: Marshaler.marshal_typed(marshaler, source, type_code, object_marshaler)
        self._demarshaler_type_map[type_code] = \
            lambda demarshaler, source: Demarshaler.demarshal_typed(
                demarshaler, source, demarshaling_factory, demarshaling_initializer)

    def marshal(self, source: any) -> any:
        return Marshaler(self._config, self._marshaler_type_map, source)()

    def demarshal(self, source: any) -> any:
        return Demarshaler(self._config, self._demarshaler_type_map, source)()

    @staticmethod
    def _initialize_tuple(demarshaler: Demarshaler, source: dict, result: tuple) -> None:
        values = demarshaler.demarshal(source[''])

        ref_count_ptr = ctypes.POINTER(ctypes.c_ssize_t)
        staticmethod(ctypes.pythonapi.PyGILState_Ensure)()
        try:
            py_object_target = ctypes.py_object(result)
            ref_count = ctypes.cast(id(result), ref_count_ptr).contents.value
            for _ in range(ref_count - 1):
                staticmethod(ctypes.pythonapi.Py_DecRef)(py_object_target)
            try:
                for i, v in values:
                    if staticmethod(ctypes.pythonapi.PyTuple_SetItem)(
                            py_object_target, ctypes.c_ssize_t(i), ctypes.py_object(v)):
                        raise SystemError('Tuple mutation failed.')
            finally:
                for _ in range(ref_count):
                    staticmethod(ctypes.pythonapi.Py_IncRef)(py_object_target)
        finally:
            staticmethod(ctypes.pythonapi.PyGILState_Release)()
