from typing import Callable, Union, Any



TypeCode = Union[int, str]
AtomicData = Union[int, float, str]
# MarshaledData = Union[AtomicData, dict, list]

ObjectMarshaler = Callable[['Marshaler', Any], Any]
ReferenceRegister = Callable[['Demarshaler', Any], Any]
ObjectDemarshaler = Callable[['Demarshaler', Any], Any]
DemarshalingFactory = Callable[['Demarshaler', Any], Any]
DemarshalingInitializer = Callable[['Demarshaler', Any, Any], Any]


PostDemarshalSetter = Callable[[Any], Any]
PostDemarshalListener = Callable[[], Any]

from lmarshal.src.demarshaler import Demarshaler
from lmarshal.src.marshaler import Marshaler