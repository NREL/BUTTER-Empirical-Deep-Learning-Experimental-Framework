from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
)
from enum import Enum


class AttributeValueType(Enum):
    Null = (0, type(None))
    Bool = (1, bool)
    Integer = (2, int)
    Float = (3, float)
    String = (4, str)
    JSON = (5, list)

    def __init__(
        self,
        type_code: int,
        python_type: Type,
        # sql_column: Optional[str],
        # sql_type: str,
    ) -> None:
        self.type_code: int = type_code
        self.python_type: Type = python_type
        # self.sql_column: Optional[str] = sql_column
        # self.sql_type: str = sql_type


_attribute_type_code_map = tuple(vt for vt in AttributeValueType)


def get_attribute_value_type_for_type_code(type_code: int) -> AttributeValueType:
    return _attribute_type_code_map[type_code]


_attribute_type_map = {vt.python_type: vt for vt in AttributeValueType}


def get_attribute_value_type_for_type(type_: Type) -> AttributeValueType:
    return _attribute_type_map.get(type_, AttributeValueType.JSON)


def get_attribute_value_type_for_value(value: Any) -> AttributeValueType:
    return get_attribute_value_type_for_type(type(value))
