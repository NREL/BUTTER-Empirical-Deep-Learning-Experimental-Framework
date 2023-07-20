from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.identifiable import Identifiable
from dmp.postgres_interface.element.column import Column

from dmp.postgres_interface.element.column_group import ColumnGroup


class Table(ColumnGroup, Identifiable):
    _name: str

    def __init__(self, name: str) -> None:
        self._name = name

        super().__init__(
            *(value for key, value in vars(self).items() if isinstance(value, Column))
        )

    @property
    def name(self) -> str:
        return self._name

    def get_index_of(self, key: Column) -> int:
        if self._index is None:
            self._index = {column: i for i, column in enumerate(self._columns)}
        return self._index[key]

    def get_column(self, index: int) -> Column:
        return self._columns[index]

    @property
    def columns(self) -> Sequence["Column"]:
        raise NotImplementedError()
