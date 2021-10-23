from typing import Callable, Union

from lmarshal.demarshaler import Demarshaler
from lmarshal.marshaler import Marshaler

TypeCode = Union[int, str]
AtomicData = Union[int, float, str]
# MarshaledData = Union[AtomicData, dict, list]

ObjectMarshaler = Callable[[Marshaler, any], any]
ReferenceRegister = Callable[[Demarshaler, any], any]
ObjectDemarshaler = Callable[[Demarshaler, any], any]
ObjectDemarshalingFactory = Callable[[Demarshaler, any], any]
ObjectDemarshalingInitializer = Callable[[Demarshaler, any, any], any]


PostDemarshalSetter = Callable[[any], any]
PostDemarshalListener = Callable[[], any]
