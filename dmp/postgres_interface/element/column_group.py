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

    def __getitem__(self, key: Union[Column, int]) -> Union[Column, int]:
        if isinstance(key, Column):
            if self._index is None:
                self._index = {column: i for i, column in enumerate(self._columns)}
            return self._index[key]
        return self._columns[key]

    @property
    def columns(self) -> Sequence[Column]:
        return self._columns
