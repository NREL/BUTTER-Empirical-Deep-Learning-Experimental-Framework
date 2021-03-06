from typing import Callable, Union


TypeCode = Union[int, str]
AtomicData = Union[int, float, str]
# MarshaledData = Union[AtomicData, dict, list]

ObjectMarshaler = Callable[['Marshaler', any], any]
ReferenceRegister = Callable[['Demarshaler', any], any]
ObjectDemarshaler = Callable[['Demarshaler', any], any]
DemarshalingFactory = Callable[['Demarshaler', any], any]
DemarshalingInitializer = Callable[['Demarshaler', any, any], any]


PostDemarshalSetter = Callable[[any], any]
PostDemarshalListener = Callable[[], any]
