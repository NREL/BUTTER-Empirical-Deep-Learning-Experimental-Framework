from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, get_args
from enum import Enum
import simplejson
import psycopg2
from psycopg2 import sql
from dmp.marshaling import marshal
from pprint import pprint

psycopg2.extras.register_uuid()
psycopg2.extras.register_default_json(loads=simplejson.loads,
                                      globally=True)  # type: ignore
psycopg2.extras.register_default_jsonb(loads=simplejson.loads,
                                       globally=True)  # type: ignore
psycopg2.extensions.register_adapter(dict,
                                     psycopg2.extras.Json)  # type: ignore


class ParameterValueType(Enum):
    Null = (0, None, None, 'NULL')
    Bool = (1, bool, 'value_bool', 'boolean')
    Integer = (2, int, 'value_int', 'bigint')
    Float = (3, float, 'value_float', 'double precision')
    String = (4, str, 'value_str', 'text')
    JSON = (5, list, 'value_json', 'jsonb')

    def __init__(
        self,
        type_code: int,
        python_type: Type,
        sql_column: Optional[str],
        sql_type: str,
    ) -> None:
        self.type_code: int = type_code
        self.python_type: Type = python_type
        self.sql_column: Optional[str] = sql_column
        self.sql_type: str = sql_type


_parameter_type_code_map = tuple(pvt for pvt in ParameterValueType)


def get_parameter_value_type_for_type_code(
        type_code: int) -> ParameterValueType:
    return _parameter_type_code_map[type_code]


_parameter_type_map = {pvt.python_type: pvt for pvt in ParameterValueType}


def get_parameter_value_type_for_type(type_: Type) -> ParameterValueType:
    return _parameter_type_map.get(type_, ParameterValueType.JSON)


def get_parameter_value_type_for_value(value: Any) -> ParameterValueType:
    return get_parameter_value_type_for_type(type(value))


ComparableValue = Union[None, bool, int, float, str]


@dataclass
class Parameter():
    id_: int
    kind: str
    value_type: ParameterValueType
    comparable_value: ComparableValue
    actual_value: Any


def _make_column_identifier_seq(
    source: List[Tuple[str, str]], ) -> List[Tuple[sql.Identifier, str]]:
    return [(sql.Identifier(column_name), column_type)
            for column_name, column_type in source]


class PostgresParameterMap:

    _parameter_table: sql.Identifier
    _select_parameter: sql.Composed
    _kind_type_value_map: Dict[str, Dict[ParameterValueType, Dict[
        ComparableValue, Parameter]]]  # {kind:{type:{comparable_value:param}}}
    _kind_value_map: Dict[str,
                          Dict[ComparableValue,
                               Parameter]]  # {kind:{comparable_value:param}}
    _id_map: Dict[int, Parameter]  # {id:param}

    _id_columns: List[Tuple[sql.Identifier,
                            str]] = _make_column_identifier_seq([
                                ('parameter_id', 'integer'),
                            ])

    _index_columns: List[Tuple[sql.Identifier,
                               str]] = _make_column_identifier_seq([
                                   ('value_type', 'smallint'),
                                   ('kind', 'text'),
                               ])

    _value_columns: List[Tuple[sql.Identifier,
                               str]] = _make_column_identifier_seq([
                                   (
                                       parameter_type.sql_column,
                                       parameter_type.sql_type,
                                   ) for parameter_type in ParameterValueType
                                   if parameter_type.sql_column is not None
                               ])

    # _column_names: List[str]

    _id_column: int = 0
    _type_column: int = 1
    _kind_column: int = 2
    _column_map: Dict[Optional[str], int]
    _type_to_column_map: Dict[ParameterValueType, int]

    _value_types: Sequence[Type] = get_args(ComparableValue) + (dict, )

    _matching_clause: sql.Composable
    _casting_clause: sql.Composable
    _key_columns: sql.Composable
    _query_values_key_columns: sql.Composable

    # _column_map : Dict[str, int]
    # _value_type_to_column_map : Dict[ParameterValueType, int]

    def __init__(
        self,
        cursor,
        parameter_table='parameter2',
    ) -> None:
        super().__init__()

        self._kind_type_value_map = {}
        self._kind_value_map = {}
        self._id_map = {}

        self._parameter_table = sql.Identifier(parameter_table)

        index_and_value_columns = self._index_columns + self._value_columns
        all_columns = self._id_columns + index_and_value_columns

        self._column_map = {
            column_name.string: i
            for i, (column_name, column_type) in enumerate(all_columns)
        }

        self._type_to_column_map = {  # type: ignore
            parameter_type: self._column_map.get(parameter_type.sql_column,
                                                 None)
            for parameter_type in ParameterValueType
        }

        self._matching_clause = sql.SQL(' and ').join((sql.SQL(
            "{parameter_table}.{column_name} IS NOT DISTINCT FROM t.{column_name}"
        ).format(
            parameter_table=self._parameter_table,
            column_name=column_name,
        ) for column_name, column_type in index_and_value_columns))

        self._casting_clause = sql.SQL(', ').join(
            (sql.SQL("{}::{}").format(column_name, sql.SQL(column_type))
             for column_name, column_type in index_and_value_columns))

        self._key_columns = sql.SQL(',').join(
            (column_name
             for column_name, column_type in index_and_value_columns))

        self._query_values_key_columns = sql.SQL(', ').join(
            (sql.SQL('query_values.{column_name}').format(
                column_name=column_name, )
             for column_name, column_type in index_and_value_columns))

        cursor.execute(
            sql.SQL("SELECT {columns} FROM {parameter_table}").format(
                columns=sql.SQL(',').join(
                    (column_name for column_name, column_type in all_columns)),
                parameter_table=self._parameter_table,
            ))

        for row in cursor.fetchall():
            value_type = get_parameter_value_type_for_type_code(
                row[self._type_column])
            value_column = self._type_to_column_map[value_type]
            database_value = row[value_column]
            value = self._recover_value_from_database(
                value_type,
                database_value,
            )

            self._register_parameter(
                Parameter(
                    row[self._id_column],
                    row[self._kind_column],
                    value_type,
                    database_value,
                    value,
                ))

    def to_parameter_id(self, kind: str, value: Any, cursor=None) -> int:
        value_type = get_parameter_value_type_for_value(value)
        comparable_value = self._make_comparable_value(value_type, value)

        try:
            return self._kind_type_value_map[kind][value_type][
                comparable_value].id_
        except KeyError:
            if cursor is None:
                raise KeyError(
                    f'Unable to translate parameter {kind} : {value}.')

            typed_values = [None] * (len(_parameter_type_code_map) - 1)
            typed_values[value_type.type_code - 1] = value

            cursor.execute(
                sql.SQL("""
WITH query_values as (
    SELECT 
        (   SELECT parameter_id
            FROM {parameter_table}
            WHERE {matching_clause}
            LIMIT 1
        ) parameter_id,
        *
    FROM
    (   SELECT {casting_clause}
        FROM (VALUES %s) AS t ({key_columns})
    ) t
),
inserted as (
    INSERT INTO {parameter_table} AS p ({key_columns})
    SELECT {key_columns}
    FROM query_values
    WHERE parameter_id IS NULL
    ON CONFLICT DO NOTHING
    RETURNING
        parameter_id, 
        {key_columns}
)
SELECT parameter_id, {key_columns} from query_values WHERE parameter_id IS NOT NULL
UNION ALL
SELECT * from inserted
;""").format(parameter_table=self._parameter_table,
             matching_clause=self._matching_clause,
             casting_clause=self._casting_clause,
             key_columns=self._key_columns,
             query_values_key_columns=self._query_values_key_columns),
                ((value_type.value, kind, *(typed_values)), ))
            result = cursor.fetchone()

            if result is not None and result[0] is not None:
                self._register_parameter(
                    Parameter(
                        int(result[0]),
                        kind,
                        value_type,
                        comparable_value,
                        value,
                    ))
            return self.to_parameter_id(kind, value, None)  # retry

    def get_all_kinds(self) -> Sequence[str]:
        return tuple(self._kind_type_value_map.keys())

    # def get_all_parameters_for_kind(self, kind) -> Sequence[Parameter]:
    #     return tuple(self._kind_value_map[kind].values())

    # def get_all_ids_for_kind(self, kind) -> Sequence[int]:
    #     return tuple(self._parameter_to_id_map[kind].values())

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
        id_: int,
        cursor=None,
    ) -> Parameter:
        return self._id_map[id_]

    def parameter_from_ids(
        self,
        ids: Iterable[int],
        cursor=None,
    ) -> List[Parameter]:
        return [self.parameter_from_id(e, cursor) for e in i]

    def _register_parameter(
        self,
        parameter: Parameter,
    ) -> None:
        self._kind_type_value_map.setdefault(parameter.kind, {}).setdefault(
            parameter.value_type, {})[parameter.comparable_value] = parameter
        self._kind_value_map.setdefault(
            parameter.kind, {})[parameter.comparable_value] = parameter
        self._id_map[parameter.id_] = parameter

    def _make_comparable_value(
        self,
        value_type: ParameterValueType,
        value: Any,
    ) -> Any:
        if value_type == ParameterValueType.JSON:
            return marshal.marshal(value)
        return simplejson.dumps(value)

    def _recover_value_from_database(
        self,
        value_type: ParameterValueType,
        database_value: Any,
    ) -> Any:
        if value_type == ParameterValueType.JSON:
            return marshal.demarshal(database_value)
        return database_value
