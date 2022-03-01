from typing import Dict


class PostgresParameterMap:

    _parameter_to_id_map: Dict[any, int]
    _id_to_parameter_map: Dict[int, any]

    key_columns = """kind, 
bool_value, 
integer_value, 
real_value, 
string_value"""

    select_parameter = f"""
SELECT id, {key_columns}
FROM 
    parameter """

    def __init__(self,
                 cursor,
                 ) -> None:
        super().__init__()

        self._parameter_to_id_map = {}
        self._id_to_parameter_map = {}

        result = cursor.execute(self.select_parameter)
        for row in result.fetchall():
            self._register_parameter(row)

    def to_parameter_id(self, kind, value, cursor=None):
        try:
            return self._parameter_to_id_map[kind][value]
        except KeyError:
            if cursor is not None:
                typed_values = self._make_typed_values(kind, value)
                result = cursor.execute(f"""
WITH i as (
    INSERT INTO parameter (
        {self.key_columns}
        )
        VALUES(%s)
    ON CONFLICT DO NOTHING
)
{self.select_parameter}
WHERE
    kind = %s and
    bool_value IS NOT DISTINCT FROM %s and
    integer_value IS NOT DISTINCT FROM %s and
    real_value IS NOT DISTINCT FROM (%s)::real and
    string_value IS NOT DISTINCT FROM %s
;""", (
                    ((kind, *typed_values),),
                    kind,
                    *typed_values,
                )
                ).fetchone()

                if result is not None:
                    self._register_parameter(result)
                    return self.to_parameter_id(kind, value, None)
            raise KeyError(f'Unable to translate parameter {kind} : {value}.')

    def to_parameter_ids(self, kvl, cursor=None):
        return [self.to_parameter_id(kind, value, cursor)
                for kind, value in kvl]

    def parameter_from_id(self, i, cursor=None):
        if isinstance(i, list):
            return [self.parameter_from_id(e, cursor) for e in i]
        return self._id_to_parameter_map[i]

    def parameter_value_from_id(self, parameter_id, cursor=None):
        try:
            return self.parameter_value_from_id[parameter_id]
        except KeyError:
            if cursor is not None:
                result = cursor.execute(f"""
{self.select_parameter}
WHERE id = %s
;""", (id,)).fetchone()
                if result is not None:
                    self._register_parameter(result)
                    return self.parameter_value_from_id(parameter_id, None)
            raise KeyError(f'Unknown parameter id {parameter_id}.')

    def _register_parameter(self, row):
        parameter_id = row[0]
        kind = row[1]
        value = next(
            (v for v in (row[i + 2] for i in range(4)) if v is not None),
            None)

        self._id_to_parameter_map[parameter_id] = (kind, value)

        if kind not in self._parameter_to_id_map:
            self._parameter_to_id_map[kind] = {}
        self._parameter_to_id_map[kind][value] = parameter_id

    def _make_typed_values(self, value):
        typed_values = [
            value if isinstance(value, t) else None
            for t in [bool, int, float, str]
        ]
        if sum((v is not None for v in typed_values)) != 1:
            raise ValueError('Value is not a supported parameter type.')
        return typed_values
