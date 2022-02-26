from typing import Dict


class PostgresParameterMap:

    _parameter_to_id_map: Dict[any, int]
    _id_to_parameter_map: Dict[int, any]

    def __init__(self,
                 cursor,
                 ) -> None:
        super().__init__()

        parameter_to_id_map = {}
        id_to_parameter_map = {}

        result = cursor.execute("""
SELECT id, kind, bool_value, integer_value, real_value, string_value from parameter;""")
        for row in result.fetchall():
            parameter_id = row[0]
            kind = row[1]
            value = None
            for i in range(4):
                if value is not None:
                    break
                value = row[i]

            if kind not in parameter_to_id_map:
                parameter_to_id_map[kind] = {}

            parameter_to_id_map[kind][value] = parameter_id
            id_to_parameter_map[parameter_id] = (kind, value)

        self._parameter_to_id_map = parameter_to_id_map
        self._id_to_parameter_map = id_to_parameter_map

    def to_parameter_id(self, kind, value, create_if_missing = False):
        try:
            return self.parameter_to_id_map[kind][value]
        except KeyError:
            pass

    def to_parameter_ids(self, kvl):
        return [self._to_parameter_id(kind, value) for kind, value in kvl]

    def parameter_from_id(self, i):
        if isinstance(i, list):
            return [self._parameter_from_id(e) for e in i]
        return self.id_to_parameter_map[i]

    def parameter_value_from_id(self, i):
        return self._parameter_from_id[1]
