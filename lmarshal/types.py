from typing import Callable, Union

from lmarshal.demarshaler import Demarshaler
from lmarshal.marshaler import Marshaler

TypeCode = Union[int, str]
AtomicData = Union[int, float, str]
# MarshaledData = Union[AtomicData, dict, list]

ObjectMarshaler = Callable[[Marshaler, any], any]
ObjectDemarshaler = Callable[[Demarshaler, any], (any, bool)]

PostDemarshalSetter = Callable[[any], any]
PostDemarshalListener = Callable[[], any]
