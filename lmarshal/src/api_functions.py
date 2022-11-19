class APIFunctions:
    PyGILState_Ensure = staticmethod(ctypes.pythonapi.PyGILState_Ensure)
    PyGILState_Release = staticmethod(ctypes.pythonapi.PyGILState_Release)
    Py_DecRef = staticmethod(ctypes.pythonapi.Py_DecRef)
    Py_IncRef = staticmethod(ctypes.pythonapi.Py_IncRef)
    PyTuple_SetItem = staticmethod(ctypes.pythonapi.PyTuple_SetItem)