from typing import Dict
from uuid import UUID
from dmp.logging.postgres_parameter_map import PostgresParameterMap
from dmp.logging.result_logger import ResultLogger


from jobqueue.cursor_manager import CursorManager

from psycopg2 import sql


class PostgresResultLogger(ResultLogger):
    _credentials: Dict[str, any]

    _parameter_map: PostgresParameterMap
    _experiment_table: sql.Identifier
    _run_table: sql.Identifier
    _run_settings_table: sql.Identifier

    def __init__(self,
                 credentials: Dict[str, any],
                 experiment_table: str = 'experiment',
                 run_table: str = 'run',
                 run_settings_table: str = 'run_settings',
                 ) -> None:
        super().__init__()
        self._credentials = credentials
        self._experiment_table = sql.Identifier(experiment_table)
        self._run_table = sql.Identifier(run_table)
        self._run_settings_table = sql.Identifier(run_settings_table)

        # initialize parameter map
        with CursorManager(self._credentials) as cursor:
            self._parameter_map = PostgresParameterMap(cursor)

    def log(
        self,
        run_id: UUID,
        experiment_parameters: Dict,
        run_parameters: Dict,
        run_values: Dict,
        result: Dict,
    ) -> None:
        with CursorManager(self._credentials) as cursor:
            # get sorted parameter ids
            run_parameter_ids = \
                self._parameter_map.to_sorted_parameter_ids(
                    run_parameters,
                    cursor
                )

            experiment_parameter_ids = \
                self._parameter_map.to_sorted_parameter_ids(
                    experiment_parameters,
                    cursor
                )

            # get experiment_id
            run_columns = sorted(list(run_values.keys()))
            result_columns = sorted(list(result.keys()))

            # create table run_settings
            # (
            #     run_id uuid NOT NULL PRIMARY KEY,
            #     experiment_id integer,
            #     record_timestamp integer DEFAULT ((date_part('epoch'::text, CURRENT_TIMESTAMP) - (1600000000)::double precision))::integer,
            #     parameters smallint[],
            #     seed bigint,
            #     save_every_epochs smallint
            # );

            cursor.execute(
                sql.SQL("""
WITH v as (
    SELECT
        job_id::uuid,
        experiment_parameters::smallint[] experiment_parameters,
        run_parameters::smallint[] run_parameters,
        {run_columns},
        {result_columns}
    FROM
        (VALUES %s ) AS t (
            job_id,
            experiment_parameters, 
            run_parameters,
            {run_columns},
            {result_columns})
),
i as (
    INSERT INTO {experiment_table} (experiment_parameters)
    SELECT experiment_parameters from v
    ON CONFLICT DO NOTHING
    RETURNING experiment_id, experiment_parameters
    UNION ALL
    SELECT experiment_id, experiment_parameters 
    FROM
        v,
        experiment e
    WHERE
        e.experiment_parameters = v.experiment_parameters
),
x as (
    SELECT 
        v.*,
        i.experiment_id experiment_id
    FROM
        v,
        i
    WHERE
        i.experiment_parameters = v.experiment_parameters
),
s as (
    INSERT INTO {run_settings_table} (
        job_id,
        experiment_id,
        run_parameters,
        {run_columns})
    SELECT 
        job_id,
        experiment_id,
        run_parameters,
        {run_columns}
    FROM
        x
    ON CONFLICT DO NOTHING
)
INSERT INTO {run_table} (
    job_id,
    experiment_id, 
    parameters, 
    {result_columns}
    )
SELECT 
    job_id,
    experiment_id,
    run_parameters,
    {result_columns}
FROM
    x
ON CONFLICT DO NOTHING
;"""
                        ).format(
                            run_columns=sql.SQL(',').join(
                                map(sql.Identifier, run_columns)),
                            result_columns=sql.SQL(',').join(
                                map(sql.Identifier, result_columns)),
                            experiment_table=self._experiment_table,
                            run_table=self._run_table,
                            run_settings_table=self._run_settings_table,
                ),
                ((
                    run_id,
                    experiment_parameter_ids,
                    run_parameter_ids,
                    *(run_values[c] for c in run_columns),
                    *(result[c] for c in result_columns),
                ),))
