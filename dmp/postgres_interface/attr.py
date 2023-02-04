
from dataclasses import dataclass
from typing import Any, Sequence, Type, Union, get_args

from dmp.postgres_interface.attribute_value_type import AttributeValueType


ComparableValue = Union[None, bool, int, float, str]
value_types: Sequence[Type] = get_args(ComparableValue) + (dict, )

@dataclass(frozen=True, slots=True)
class Attr():
    attr_id: int
    kind: str
    value_type: AttributeValueType
    comparable_value: ComparableValue
    value: Any