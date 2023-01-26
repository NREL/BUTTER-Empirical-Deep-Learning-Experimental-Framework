from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
import itertools
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb, Json
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder


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
    def column(self) -> str:
        if len(self._columns) != 1:
            raise ValueError(
                f'Called column property on ColumnGroup with {len(self._columns)} columns.'
            )
        return self._columns[0]

    @property
    def types(self) -> Sequence[str]:
        return self._types

    @property
    def columns_and_types(self) -> Sequence[Tuple[str, str]]:
        return tuple(zip(self._columns, self._types))

    def column_as_group(self, key: Union[str, int]) -> 'ColumnGroup':
        index = None
        if isinstance(key, int):
            index = key
        else:
            index = self._index[key]

        return ColumnGroup([
            (self._columns[index], self._types[index]),
        ])

    @property
    def identifiers(self) -> Sequence[Identifier]:
        return tuple((Identifier(name) for name in self._columns))

    @property
    def identifier(self) -> Identifier:
        return Identifier(self.column)

    @property
    def columns_sql(self) -> Composed:
        return sql_comma.join(self.identifiers)

    @property
    def casting_sql(self) -> Composed:
        return sql_comma.join((
            SQL('{}::{}').format(Identifier(name), SQL(type))  # type: ignore
            for name, type in self.columns_and_types))

    @property
    def placeholders(self) -> Composed:
        return sql_comma.join([sql_placeholder] * len(self._columns))

    def of(self, table_name: Identifier) -> Composed:
        return sql_comma.join((
            SQL('{}.{}').format(table_name, column)  # type: ignore
            for column in self.identifiers))

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
