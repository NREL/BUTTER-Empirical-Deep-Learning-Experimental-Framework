from dataclasses import dataclass
from typing import Any, Sequence, Type, Union, get_args

from dmp.postgres_interface.attribute_value_type import AttributeValueType

ComparableValue = Union[None, bool, int, float, str]
value_types: Sequence[Type] = get_args(ComparableValue) + (dict,)


# @dataclass(frozen=True, slots=True)
class Attr:
    __slots__ = (
        "attr_id",
        "kind",
        "value_type",
        "comparable_value",
        "value",
    )

    attr_id: int
    kind: str
    value_type: AttributeValueType
    comparable_value: ComparableValue
    value: Any

    def __init__(
        self,
        attr_id: int,
        kind: str,
        value_type: AttributeValueType,
        comparable_value: ComparableValue,
        value: Any,
    ) -> None:
        self.attr_id: int = attr_id
        self.kind: str = kind
        self.value_type: AttributeValueType = value_type
        self.comparable_value: ComparableValue = comparable_value
        self.value: Any = value
