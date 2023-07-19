from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from dataclasses import dataclass
from psycopg.sql import Identifier, SQL, Composed, Literal
from dmp.postgres_interface.element.identifiable import Identifiable
from dmp.postgres_interface.element.a_column_group import AColumnGroup


@dataclass(eq=True, frozen=True)
class Column(AColumnGroup, Identifiable):
    _name: str
    type_name: str

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> Sequence["Column"]:
        return (self,)

    def get_index_of(self, key: Column) -> int:
        return 0

    def get_column(self, index: int) -> Column:
        return self

    def column(self, key: Union[str, int]) -> Column:
        if key != 0 and key != self.name:
            raise KeyError()
        return self
