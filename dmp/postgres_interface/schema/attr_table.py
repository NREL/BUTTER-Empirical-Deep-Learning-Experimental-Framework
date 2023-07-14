from dataclasses import dataclass
from itertools import chain
from typing import Mapping
from dmp.postgres_interface.attribute_value_type import AttributeValueType
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


@dataclass(frozen=True)
class AttrTable(Table):
    name: str = "attr"
    attr_id: Column = Column("attr_id", "integer")
    value_type: Column = Column("value_type", "smallint")
    kind: Column = Column("kind", "text")
    value_bool: Column = Column("value_bool", "boolean")
    value_int: Column = Column("value_int", "bigint")
    value_float: Column = Column("value_float", "double precision")
    value_str: Column = Column("value_str", "text")
    digest: Column = Column("digest", "uuid")
    value_json: Column = Column("value_json", "jsonb")

    @property
    def index(self) -> ColumnGroup:
        return ColumnGroup(
            self.value_type,
            self.kind,
        )

    @property
    def value(self) -> ColumnGroup:
        return ColumnGroup(
            self.value_bool,
            self.value_int,
            self.value_float,
            self.value_str,
            self.value_json,
        )

    @property
    def all_except_id(self) -> ColumnGroup:
        return ColumnGroup(
            self.value_type,
            self.kind,
            self.value_bool,
            self.value_int,
            self.value_float,
            self.value_str,
            self.digest,
            self.value_json,
        )

    @property
    def all(self) -> ColumnGroup:
        return ColumnGroup(
            self.attr_id,
            self.value_type,
            self.kind,
            self.value_bool,
            self.value_int,
            self.value_float,
            self.value_str,
            self.value_json,
            self.digest,
        )

    @property
    def attribute_value_type_map(self) -> Mapping[AttributeValueType, Column]:
        return {
            attribute_type: column
            for attribute_type, column in zip(
                AttributeValueType, chain((None,), self.value)
            )
            if attribute_type is not AttributeValueType.Null
        }  # type: ignore
