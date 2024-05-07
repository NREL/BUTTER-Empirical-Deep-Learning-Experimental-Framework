import ctypes


class APIFunctions:
    PyGILState_Ensure = staticmethod(ctypes.pythonapi.PyGILState_Ensure)
    PyGILState_Release = staticmethod(ctypes.pythonapi.PyGILState_Release)
    Py_DecRef = staticmethod(ctypes.pythonapi.Py_DecRef)
    Py_IncRef = staticmethod(ctypes.pythonapi.Py_IncRef)
    PyTuple_SetItem = staticmethod(ctypes.pythonapi.PyTuple_SetItem)


def marshal_tuple(marshaller: 'Marshaler', source: tuple) -> dict:
    return {
        marshaller._config.flat_dict_key:
        Marshaler.marshal_list(marshaller, (e for e in source))
    }


def demarshal_tuple(demarshaler: 'Demarshaler', source: dict) -> tuple:
    return (None, ) * len(source[demarshaler._config.flat_dict_key])


def initialize_tuple(
    demarshaler: 'Demarshaler',
    source: dict,
    result: tuple,
) -> None:
    values = demarshaler.demarshal(source[demarshaler._config.flat_dict_key])
    if not isinstance(values, list):
        raise TypeError(
            f'Found a {type(values)} when expecting list of values while demarshaling a tuple.'
        )

    ref_count_ptr = ctypes.POINTER(ctypes.c_ssize_t)
    APIFunctions.PyGILState_Ensure()
    try:
        py_object_target = ctypes.py_object(result)
        ref_count = \
            ctypes.cast(id(result), ref_count_ptr).contents.value - 1
        for _ in range(ref_count):
            APIFunctions.Py_DecRef(py_object_target)

        try:
            for i, v in enumerate(values):
                value_py_object = ctypes.py_object(v)
                APIFunctions.Py_IncRef(value_py_object)
                if APIFunctions.PyTuple_SetItem(
                        py_object_target,
                        ctypes.c_ssize_t(i),
                        value_py_object,
                ):
                    raise SystemError('Tuple mutation failed.')
        finally:
            for _ in range(ref_count):
                APIFunctions.Py_IncRef(py_object_target)
    finally:
        APIFunctions.PyGILState_Release()


from .demarshaler import Demarshaler
from .marshaler import Marshaler