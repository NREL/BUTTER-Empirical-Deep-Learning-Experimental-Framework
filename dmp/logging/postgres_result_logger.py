from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Union
from dmp.logging.result_logger import ResultLogger

from dmp.task.task import Task

from jobqueue.cursor_manager import CursorManager

import psycopg2
from psycopg2 import sql


class PostgresResultLogger(ResultLogger):
    _credentials: Dict[str, any]
    _parameter_map: Dict[any, int]

    def __init__(self,
                 credentials: Dict[str, any],
                 ) -> None:
        super().__init__()
        self._credentials = credentials
        self._parameter_to_id_map, self._id_to_parameter_map = \
            self._make_parameter_maps()

    def log(
        self,
        experiment_parameters: Dict,
        run_parameters: Dict,
        result: Dict,
    ) -> None:
        with CursorManager(self._credentials) as cursor:
            experiment_parameter_ids = \
                self.get_parameter_ids(experiment_parameters)

            # get experiment_id
            # if experiment doesn't exist, create it using experiment_parameters
            pass

            # insert run_parameters and result into run table

        pass

    def get_parameter_ids(self, cursor, parameters):
        return [self.get_parameter_id(k, v) for k, v in parameters.items()]

    def get_parameter_id(self, cursor, key, value):
        # get parameter id from table
        # if it doesn't exist, create it
        cursor.execute(sql.SQL("""
INSERT INTO parameter (kind, value) VALUES(%s) ON CONFLICT(value) DO NOTHING;
"""), [(key, value)])

        parameter_id = cursor.execute(sql.SQL("""
SELECT id FROM parameter WHERE 
    int_value = %s AND
    real_value = %s AND
    string_value = %s;""")).fetchone()

        return parameter_id

    def _make_parameter_maps(self):
        df = pd.read_sql(
            f'''SELECT id, kind, real_value, integer_value, string_value from parameter''',
            db.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False,
            params=())
        df['value'] = df.real_value.combine_first(df.integer_value).\
            combine_first(df.string_value)
        print(df)

        parameter_to_id_map = {kind : {} for kind in df['kind'].unique().tolist()}
        id_to_parameter_map = {}
        for index, row in df.iterrows():
            i = row['id']
            k = row['kind']
            v = row['value'] 
            
            parameter_to_id_map[k][v] = i
            id_to_parameter_map[i] = (k, v)
        return parameter_to_id_map, id_to_parameter_map
        
    def _to_parameter_id(self, kind, value):
        return self.parameter_to_id_map[kind][value]

    def _to_parameter_ids(self, kvl):
        return [self._to_parameter_id(kind, value) for kind, value in kvl]

    def _parameter_from_id(self, i):
        if isinstance(i, list):
            return [self._parameter_from_id(e) for e in i]
        return self.id_to_parameter_map[i]

    def _parameter_value_from_id(self, i):
        return self._parameter_from_id[1]

