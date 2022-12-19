from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from psycopg2 import sql


class PostgresParameterMap:

    _parameter_table: sql.Identifier
    _select_parameter: sql.Composed
    _parameter_to_id_map: Dict[str, Dict[Any, int]]
    _id_to_parameter_map: Dict[int, Tuple[str, Any]]

    _key_columns = sql.SQL(',').join(
        map(sql.Identifier, [
            'kind',
            'bool_value',
            'integer_value',
            'real_value',
            'string_value',
        ]))

    def __init__(
        self,
        cursor,
        parameter_table='parameter_',
    ) -> None:
        super().__init__()

        self._parameter_table = sql.Identifier(parameter_table)
        self._select_parameter = \
            sql.SQL("""
SELECT id, {}
FROM {}"""
                    ).format(self._key_columns, self._parameter_table)
        self._parameter_to_id_map = {}
        self._id_to_parameter_map = {}

        cursor.execute(self._select_parameter)
        for row in cursor.fetchall():
            self._register_parameter_from_row(row)

    def to_parameter_id(self, kind: str, value: Any, cursor=None) -> int:
        try:
            return self._parameter_to_id_map[kind][value]
        except KeyError:
            if cursor is not None:
                typed_values = self._make_typed_values(value)

                cursor.execute(
                    sql.SQL("""
WITH v as (
    SELECT 
        (
            SELECT id from {_parameter_table}
            WHERE
                kind = t.kind and
                bool_value IS NOT DISTINCT FROM t.bool_value and
                integer_value IS NOT DISTINCT FROM t.integer_value and
                real_value IS NOT DISTINCT FROM (t.real_value)::real and
                string_value IS NOT DISTINCT FROM t.string_value
            LIMIT 1
        ) id,
        *
    FROM 
    (
        SELECT
        kind,
        bool_value::bool,
        integer_value::bigint,
        real_value::real,
        string_Value::text
        FROM (VALUES %s) AS t ({_key_columns})
    ) t
),
i as (
    INSERT INTO {_parameter_table} ({_key_columns})
        SELECT {_key_columns} FROM (SELECT * from v WHERE v.id IS NULL) s
    ON CONFLICT DO NOTHING
    RETURNING id, {_key_columns}
)
SELECT * from v WHERE id IS NOT NULL
UNION ALL 
SELECT * from i
;""").format(
                        _parameter_table=self._parameter_table,
                        _key_columns=self._key_columns,
                    ), ((kind, *typed_values), ))
                result = cursor.fetchone()

                if result is not None:
                    self._register_parameter(kind, value, result[0])
                    return self.to_parameter_id(kind, value, None)
            raise KeyError(f'Unable to translate parameter {kind} : {value}.')

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

    def _register_parameter_from_row(self, row) -> None:
        value = next(
            (v for v in (row[i + 2] for i in range(4)) if v is not None), None)
        self._register_parameter(row[1], value, row[0])

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

    def _make_typed_values(self, value: Any) -> List[Any]:
        value_types = [bool, int, float, str]
        typed_values = [None] * len(value_types)
        type_index = None
        for i, t in enumerate(value_types):
            if isinstance(value, t):
                typed_values[i] = value
                type_index = i
                break

        if type_index is None and value is not None:
            raise ValueError(
                f'Value is not a supported parameter type, type "{type(value)}".'
            )
        return typed_values
