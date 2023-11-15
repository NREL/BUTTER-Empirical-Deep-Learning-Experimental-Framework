from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from itertools import chain
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column


class ColumnGroup(AColumnGroup):
    _columns: Sequence[Column]
    _index: Optional[Dict[Column, int]]

    def __init__(self, *groups: AColumnGroup) -> None:
        self._columns = tuple(chain(*groups))
        self._index = None

    def get_index_of(self, key: Column) -> int:
        if self._index is None:
            self._index = {column: i for i, column in enumerate(self._columns)}
        print(self._index)
        return self._index[key]

    def get_column(self, index: int) -> Column:
        return self._columns[index]

    @property
    def columns(self) -> Sequence[Column]:
        return self._columns
