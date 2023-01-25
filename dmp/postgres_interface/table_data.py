from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from psycopg.sql import Identifier, SQL, Composed, Literal
from dmp.postgres_interface.column_group import ColumnGroup

class TableData():
    _name: str
    _groups: Dict[str, ColumnGroup]

    def __init__(
        self,
        name: str,
        column_groups: Dict[str, ColumnGroup],
    ) -> None:
        self._name = name
        self._groups = column_groups

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> Identifier:
        return Identifier(self._name)

    def __getitem__(self, group_name: str) -> ColumnGroup:
        return self._groups[group_name]