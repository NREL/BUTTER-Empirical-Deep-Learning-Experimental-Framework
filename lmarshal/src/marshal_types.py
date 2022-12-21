from typing import Callable, Dict, Iterable, Tuple, Union, Any



TypeCode = Union[int, str]
AtomicData = Union[int, float, str]
# MarshaledData = Union[AtomicData, dict, list]

RawObjectMarshaler = Callable[['Marshaler', Any], Any]
ReferenceRegister = Callable[['Demarshaler', Any], Any]
RawObjectDemarshaler = Callable[['Demarshaler', Any], Any]
DemarshalingFactory = Callable[['Demarshaler', Any], Any]
DemarshalingInitializer = Callable[['Demarshaler', Any, Any], Any]

ObjectMarshaler = Callable[[Any], Dict[str, Any]]
# ObjectDemarshaler = Callable[[Any], Dict[str, Any]]

PostDemarshalSetter = Callable[[Any], Any]
PostDemarshalListener = Callable[[], Any]

from lmarshal.src.demarshaler import Demarshaler
from lmarshal.src.marshaler import Marshaler