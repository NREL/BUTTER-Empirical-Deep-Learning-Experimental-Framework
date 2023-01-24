from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
import itertools
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json
from dmp.postgres_interface.postgres_interface_common import comma_sql, placeholder_sql

class ColumnGroup():
    _columns: Sequence[str]
    _types: Sequence[str]
    _index: Dict[str, int]

    def __init__(self, columns_and_types: Iterable[Tuple[str, str]]) -> None:
        self._columns = tuple((name for name, type in columns_and_types))
        self._types = tuple((type for name, type in columns_and_types))
        self._index = {name: i for i, name in enumerate(self._columns)}

    def __getitem__(self, key: Union[str, int]) -> Union[str, int]:
        if isinstance(key, str):
            return self._index[key]
        return self._columns[key]
            

    @staticmethod
    def concatenate(groups: Iterable['ColumnGroup']) -> 'ColumnGroup':
        return ColumnGroup(
            tuple(
                itertools.chain(*(group.columns_and_types
                                  for group in groups))))

    def __add__(self, other: 'ColumnGroup') -> 'ColumnGroup':
        return self.concatenate((self, other))

   
    @property
    def columns(self) -> Sequence[str]:
        return self._columns

    @property
    def types(self) -> Sequence[str]:
        return self._types

    @property
    def columns_and_types(self) -> Iterable[Tuple[str, str]]:
        return zip(self._columns, self._types)

    @property
    def column_identifiers(self)->Sequence[Identifier]:
        return tuple((Identifier(name) for name in self._columns))

    @property
    def columns_sql(self) -> Composed:
        return comma_sql.join(self.column_identifiers)

    @property
    def casting_sql(self) -> Composed:
        return comma_sql.join((
            SQL('{}::{}').format(Identifier(name), SQL(type))  # type: ignore
            for name, type in self.columns_and_types
        ))

    @property
    def placeholders(self) -> Composed:
        return comma_sql.join([placeholder_sql] * len(self._columns))

    def columns_from(self, table_name:Identifier)->Composed:
        return comma_sql.join((
            SQL('{}.{}').format(table_name, column)  # type: ignore
            for column in self.column_identifiers
        ))

    def extract_column_values(
        self,
        source: Dict[str, Any],
    ) -> List[Any]:
        result = []
        for name, type_name in self.columns_and_types:
            value = source.pop(name, None)
            if type_name == 'jsonb' and value is not None:
                value = Jsonb(value)
            elif type_name == 'json' and value is not None:
                value = Json(value)
            else:
                pass
            result.append(value)
        return result

    