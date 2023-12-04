from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from itertools import chain
from abc import ABC, abstractproperty, abstractmethod
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.postgres_interface.element.column import Column
    from dmp.postgres_interface.element.column_group import ColumnGroup


class AColumnGroup(Iterable["Column"]):
    def __add__(self, other: AColumnGroup) -> ColumnGroup:
        from dmp.postgres_interface.element.column_group import ColumnGroup

        return ColumnGroup(self, other)

    def __getitem__(self, key: Union["Column", int]) -> Union["Column", int]:
        from dmp.postgres_interface.element.column import Column

        if isinstance(key, Column):
            return self.get_index_of(key)
        return self.get_column(key)

    @abstractmethod
    def get_index_of(self, key: Column) -> int:
        pass

    @abstractmethod
    def get_column(self, index: int) -> Column:
        pass

    def __iter__(self):
        return self.columns.__iter__()

    def __len__(self) -> int:
        return len(self.columns)

    @abstractproperty
    def columns(self) -> Sequence["Column"]:
        raise NotImplementedError()

    @property
    def names(self) -> Sequence[str]:
        return tuple((column.name for column in self.columns))

    @property
    def type_names(self) -> Sequence[str]:
        return tuple((column.type_name for column in self.columns))

    @property
    def identifiers(self) -> Sequence[Identifier]:
        return tuple((column.identifier for column in self.columns))

    @property
    def columns_sql(self) -> Composed:
        return sql_comma.join(self.identifiers)

    @property
    def casting_sql(self) -> Composed:
        return sql_comma.join(
            (
                SQL("{}::{}").format(
                    Identifier(column.name), SQL(column.type_name)  # type: ignore
                )
                for column in self.columns
            )
        )

    @property
    def placeholders(self) -> Composed:
        return sql_comma.join([sql_placeholder] * len(self.columns))

    def placeholders_for_values(self, num_values: int) -> Composed:
        return sql_comma.join([SQL("({})").format(self.placeholders)] * num_values)

    def of(self, table_name: Identifier) -> Composed:
        return sql_comma.join(
            (
                SQL("{}.{}").format(table_name, column_identifier)
                for column_identifier in self.identifiers
            )
        )

    def set_clause(self, table_name: Identifier) -> Composed:
        return sql_comma.join(
            (
                SQL("{} = {}.{}").format(
                    column.identifier, table_name, column.identifier
                )
                for column in self.columns
            )
        )

    def extract_column_values(
        self,
        source: Dict[str, Any],
        second_source: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        result = []
        second_source = {} if second_source is None else second_source
        for column in self.columns:
            value = second_source.pop(column.name, None)
            if column.name in source:
                value = source.pop(column.name)
            type_name = column.type_name
            if type_name == "jsonb" and value is not None:
                value = Jsonb(value)
            elif type_name == "json" and value is not None:
                value = Json(value)
            else:
                pass
            result.append(value)
        return result
