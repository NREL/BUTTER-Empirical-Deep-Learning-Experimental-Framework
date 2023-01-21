from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, get_args
from enum import Enum

class AttributeValueType(Enum):
    Null = (0, type(None), None, 'NULL')
    Bool = (1, bool, 'value_bool', 'boolean')
    Integer = (2, int, 'value_int', 'bigint')
    Float = (3, float, 'value_float', 'double precision')
    String = (4, str, 'value_str', 'text')
    JSON = (5, list, 'digest', 'uuid')

    def __init__(
        self,
        type_code: int,
        python_type: Type,
        sql_column: Optional[str],
        sql_type: str,
    ) -> None:
        self.type_code: int = type_code
        self.python_type: Type = python_type
        self.sql_column: Optional[str] = sql_column
        self.sql_type: str = sql_type


_attribute_type_code_map = tuple(pvt for pvt in AttributeValueType)

def get_attribute_value_type_for_type_code(
        type_code: int) -> AttributeValueType:
    return _attribute_type_code_map[type_code]


_attribute_type_map = {pvt.python_type: pvt for pvt in AttributeValueType}

def get_attribute_value_type_for_type(type_: Type) -> AttributeValueType:
    return _attribute_type_map.get(type_, AttributeValueType.JSON)


def get_attribute_value_type_for_value(value: Any) -> AttributeValueType:
    return get_attribute_value_type_for_type(type(value))
