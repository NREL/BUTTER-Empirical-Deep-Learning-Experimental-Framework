from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Type, Union
from enum import IntEnum

from psycopg2 import sql


class ParameterValueType(IntEnum):
    Null = 0
    Bool = 1
    Integer = 2
    Float = 3
    String = 4


def _make_column_identifier_seq(
    source: List[Tuple[str, str]],
) -> List[Tuple[sql.Identifier, sql.Identifier]]:
    return [(sql.Identifier(column_name), sql.Identifier(column_type))
            for column_name, column_type in source]


class PostgresParameterMap:

    _parameter_table: sql.Identifier
    _select_parameter: sql.Composed
    _parameter_to_id_map: Dict[str, Dict[Any, int]]  # {kind->{value->id}}
    _id_to_parameter_map: Dict[int, Tuple[str, Any]]  # {id->(kind,value)}

    _id_columns: List[Tuple[sql.Identifier,
                            sql.Identifier]] = _make_column_identifier_seq([
                                ('id', 'integer'),
                            ])

    _index_columns: List[Tuple[sql.Identifier,
                               sql.Identifier]] = _make_column_identifier_seq([
                                   ('value_type', 'smallint'),
                                   ('kind', 'text'),
                               ])

    _value_columns: List[Tuple[sql.Identifier,
                               sql.Identifier]] = _make_column_identifier_seq([
                                   ('bool_value', 'boolean'),
                                   ('integer_value', 'bigint'),
                                   ('float_value', 'double precision'),
                                   ('string_value', 'text'),
                               ])

    # _column_names: List[str]

    _id_column: int = 0
    _type_column: int = 1
    _kind_column: int = 2
    _type_to_column_map: Dict[int, int] = {
        ParameterValueType.Null: 3,
        ParameterValueType.Bool: 3,
        ParameterValueType.Integer: 4,
        ParameterValueType.Float: 5,
        ParameterValueType.String: 6,
    }

    _value_types: List[Type] = [type(None), bool, int, float, str]

    _matching_clause: sql.Composable
    _casting_clause: sql.Composable
    _key_columns: sql.Composable
    _query_values_key_columns: sql.Composable

    # _column_map : Dict[str, int]
    # _value_type_to_column_map : Dict[ParameterValueType, int]

    def __init__(
        self,
        cursor,
        parameter_table='parameter',
    ) -> None:
        super().__init__()

        self._parameter_table = sql.Identifier(parameter_table)
        # self._column_names = [name for name, type in self._columns]

        # self._column_map = {
        #     name : col for col, name in enumerate(self._columns)
        # }

        index_and_value_columns = self._index_columns + self._value_columns

        self._matching_clause = sql.SQL(' and ').join(
            (sql.SQL(
                "{column_name} IS NOT DISTINCT FROM t.{column_name}").format(
                    column_name=column_name, )
             for column_name, column_type in index_and_value_columns))

        self._casting_clause = sql.SQL(', ').join(
            (sql.SQL("{column_name}::{column_type}").format(
                column_name=column_name,
                column_type=column_type,
            ) for column_name, column_type in index_and_value_columns))

        self._key_columns = sql.SQL(',').join(
            (column_name
             for column_name, column_type in index_and_value_columns))

        self._query_values_key_columns = sql.SQL(', ').join(
            (sql.SQL(
                'query_values.{column_name}').format(
                    column_name=column_name, )
             for column_name, column_type in index_and_value_columns))

        self._parameter_to_id_map = {}
        self._id_to_parameter_map = {}

        cursor.execute(
            sql.SQL("SELECT {columns} FROM {parameter_table}").format(
                columns=sql.SQL(',').join(
                    (column_name for column_name, column_type in (
                        self._id_columns + index_and_value_columns))),
                parameter_table=self._parameter_table,
            ))

        for row in cursor.fetchall():
            value_type = ParameterValueType(row[self._type_column])
            value_column = self._type_to_column_map[value_type]
            self._register_parameter(
                row[self._kind_column],
                row[value_column],
                row[self._id_column],
            )

    def to_parameter_id(self, kind: str, value: Any, cursor=None) -> int:
        try:
            return self._parameter_to_id_map[kind][value]
        except KeyError:
            if cursor is None:
                raise KeyError(
                    f'Unable to translate parameter {kind} : {value}.')
            value_type, typed_values = self._make_typed_values(value)

            cursor.execute(
                sql.SQL("""
WITH query_values as (
    SELECT 
        (   SELECT id
            FROM {parameter_table}
            WHERE {matching_clause}
            LIMIT 1
        ) id,
        *
    FROM
    (   SELECT {casting_clause}
        FROM (VALUES %s) AS t ({key_columns})
    ) t
),
to_insert as (
    SELECT {key_columns}
    FROM query_values
    WHERE id IS NULL
),
inserted as (
    INSERT INTO {parameter_table} p ({key_columns})
    SELECT *
    FROM to_insert
    ON CONFLICT ({key_columns}) DO UPDATE
        SET id = p.id WHERE FALSE
    RETURNING
        id, 
        {key_columns}
),
retried AS (
    INSERT INTO {parameter_table} p ({key_columns})
    SELECT to_insert.*
    FROM
        to_insert
        LEFT JOIN inserted USING ({key_columns})
    WHERE inserted.id IS NULL
    ON CONFLICT ({key_columns}) DO UPDATE
        SET id = p.id WHERE FALSE
    RETURNING
        id,
        {key_columns}
   )
SELECT * from query_values WHERE id IS NOT NULL
UNION ALL
SELECT * from inserted WHERE id IS NOT NULL
UNION ALL 
SELECT * from retried
;""").format(
                    parameter_table=self._parameter_table,
                    key_columns=self._key_columns,
                    query_values_key_columns=self._query_values_key_columns
                ), ((value_type.value, kind, *(typed_values[1:])), ))
            result = cursor.fetchone()

            if result is not None and result[0] is not None:
                self._register_parameter(kind, value, result[0])
            return self.to_parameter_id(kind, value, None)  # retry

    def get_all_kinds(self) -> Sequence[str]:
        return tuple(self._parameter_to_id_map.keys())

    def get_all_parameters_for_kind(self, kind) -> Sequence[Any]:
        return tuple(self._parameter_to_id_map[kind].items())

    def get_all_ids_for_kind(self, kind) -> Sequence[int]:
        return tuple(self._parameter_to_id_map[kind].values())

    def to_parameter_ids(
        self,
        kvl: Union[Dict[str, Any], Iterable[Tuple[str, Any]]],
        cursor=None,
    ) -> Sequence[int]:
        if isinstance(kvl, dict):
            kvl = kvl.items()  # type: ignore
        return [
            self.to_parameter_id(kind, value, cursor) for kind, value in kvl
        ]

    def to_sorted_parameter_ids(
        self,
        kvl: Union[Dict[str, Any], Iterable[Tuple[str, Any]]],
        cursor=None,
    ) -> List[int]:
        return sorted(self.to_parameter_ids(kvl, cursor=cursor))

    def parameter_from_id(
        self,
        i: Union[List[int], int],
        cursor=None,
    ):
        if isinstance(i, list):
            return [self.parameter_from_id(e, cursor) for e in i]
        return self._id_to_parameter_map[i]

    def _register_parameter(
        self,
        kind: str,
        value: Any,
        parameter_id: int,
    ) -> None:
        self._id_to_parameter_map[parameter_id] = (kind, value)

        if kind not in self._parameter_to_id_map:
            self._parameter_to_id_map[kind] = {}
        self._parameter_to_id_map[kind][value] = parameter_id

    def _make_typed_values(
        self,
        value: Any,
    ) -> Tuple[ParameterValueType, List[Any]]:
        typed_values = [None] * len(self._value_types)
        value_type = None
        for i, t in enumerate(self._value_types):
            if isinstance(value, t):
                typed_values[i] = value
                value_type = ParameterValueType(i)
                break

        if value_type is None:
            raise ValueError(
                f'Value is not a supported parameter type, type "{type(value)}".'
            )
        return value_type, typed_values
