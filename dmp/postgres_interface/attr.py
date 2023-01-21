
from dataclasses import dataclass
from typing import Sequence, Type, Union, get_args

from dmp.postgres_interface.attribute_value_type import AttributeValueType


ComparableValue = Union[None, bool, int, float, str]
value_types: Sequence[Type] = get_args(ComparableValue) + (dict, )

@dataclass
class Attr():
    id_: int
    kind: str
    value_type: AttributeValueType
    comparable_value: ComparableValue
    # actual_value: Any