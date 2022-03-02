from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Union
from dmp.logging.postgres_parameter_map import PostgresParameterMap
from dmp.logging.result_logger import ResultLogger

from dmp.task.task import Task

from jobqueue.cursor_manager import CursorManager

import psycopg2
from psycopg2 import sql


class PostgresResultLogger(ResultLogger):
    _credentials: Dict[str, any]
    _parameter_map: PostgresParameterMap

    def __init__(self,
                 credentials: Dict[str, any],
                 ) -> None:
        super().__init__()
        self._credentials = credentials

        # initialize parameter map
        with CursorManager(self._credentials) as cursor:
            self._parameter_map = PostgresParameterMap(cursor)

    def log(
        self,
        experiment_parameters: Dict,
        run_parameters: Dict,
        result: Dict,
    ) -> None:
        with CursorManager(self._credentials) as cursor:
            # get sorted parameter ids list
            parameter_ids = sorted(
                self._parameter_map.to_parameter_ids(experiment_parameters))

            # get experiment_id
            run_columns = sorted(list(run_parameters.keys()))
            result_columns = sorted(list(result.keys()))
            columns = \
                self._make_identifier_list([
                    *run_columns,
                    *result_columns,
                ])

            result = cursor.execute(
                sql.SQL("""
WITH v as (
    SELECT
        parameters::smallint[] parameters,
        {columns}
    FROM
        (VALUES (%s), )  AS t (
            parameters, 
            {columns}
        )
),
i as (
    INSERT INTO experiment (parameters)
    SELECT parameters from v
    ON CONFLICT DO NOTHING
)
INSERT INTO run (experiment_id, parameters, {columns})
SELECT 
    e.experiment_id,
    v.*
FROM
    v,
    experiment e
WHERE
    e.parameters = v.parameters
ON CONFLICT DO NOTHING
;"""
                        ).format(columns=columns),
                ([
                    parameter_ids,
                    *(run_parameters[c]
                      for c in run_columns),
                    *(result[c] for c in result_columns),
                ],))

    def _make_identifier_list(self, identifiers):
        return sql.SQL(',').join(map(sql.Identifier, identifiers))
