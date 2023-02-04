from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from dataclasses import dataclass
from psycopg.sql import Identifier, SQL, Composed, Literal
from dmp.postgres_interface.element.identifiable import Identifiable
from dmp.postgres_interface.element.a_column_group import AColumnGroup


@dataclass(eq=True, frozen=True)
class Column(AColumnGroup, Identifiable):
    name: str
    type_name: str

    @property
    def columns(self) -> Sequence['Column']:
        return (self, )

    def __getitem__(self, key: Union['Column', int]) -> Union['Column', int]:
        if key == 0:
            return self
        if key == self:
            return 0
        raise KeyError()

    def column(self, key: Union[str, int]) -> 'Column':
        if key != 0 and key != self.name:
            raise KeyError()
        return self