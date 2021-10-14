from typing import Callable, Optional, Hashable

TypeCode = Hashable
# AtomicData = Union[int, float, str]
# MarshaledData = Union[AtomicData, dict, list]

ObjectMarshaler = Optional[Callable[[any], any]]
ObjectDemarshaler = Callable[[any], any]
