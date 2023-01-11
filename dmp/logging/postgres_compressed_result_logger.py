from pprint import pprint
from uuid import UUID
from dmp.logging.postgres_parameter_map import PostgresParameterMap
from dmp.logging.result_logger import ResultLogger

from jobqueue.cursor_manager import CursorManager

from psycopg2 import sql

from typing import Any, Dict, Iterable, Optional, Tuple, List

import simplejson
import psycopg2

from dmp.task.task_result_record import TaskResultRecord

psycopg2.extras.register_default_json(loads=simplejson.loads,
                                      globally=True)  # type: ignore
psycopg2.extras.register_default_jsonb(loads=simplejson.loads,
                                       globally=True)  # type: ignore
psycopg2.extensions.register_adapter(dict,
                                     psycopg2.extras.Json)  # type: ignore


class PostgresCompressedResultLogger(ResultLogger):
    _credentials: Dict[str, Any]
    _run_columns: List[Tuple[str, str]]
    _log_query_prefix: sql.Composed
    _log_query_suffix: sql.Composed
    _parameter_map: PostgresParameterMap

    def __init__(
        self,
        credentials: Dict[str, Any],
    ) -> None:
        super().__init__()
        self._credentials = credentials

        self._run_table: str = 'run'
        self._experiment_table: str = 'experiment'

        self._experiment_columns = [
            ('model_structure', 'bytea'),
        ]

        self._run_columns = [
            ('task_version', 'smallint'),
            ('num_nodes', 'smallint'),
            ('slurm_job_id', 'bigint'),
            ('job_id', 'uuid'),
            ('run_id', 'uuid'),
            ('num_cpus', 'smallint'),
            ('num_gpus', 'smallint'),
            ('gpu_memory', 'integer'),
            ('seed', 'bigint'),
            ('host_name', 'text'),
            ('run_history', 'bytea'),
        ]

        experiment_columns_sql, cast_experiment_columns_sql = \
            self._make_column_sql(self._experiment_columns)
        run_columns_sql, cast_run_columns_sql = \
            self._make_column_sql(self._run_columns)

        self._log_query_prefix = sql.SQL("""
WITH query_values as (
    SELECT
        (   SELECT experiment_id 
            FROM {experiment_table} e 
            WHERE e.experiment_parameters = t.experiment_parameters::::integer[]
            LIMIT 1
        ) experiment_id,
        experiment_parameters::integer[] experiment_parameters,
        experiment_attributes::integer[] experiment_attributes,
        {cast_experiment_columns},
        run_parameters::integer[] run_parameters,
        {cast_run_columns}
    FROM
        (VALUES """).format(
            cast_experiment_columns=cast_experiment_columns_sql,
            cast_run_columns=cast_run_columns_sql,
        )

        self._log_query_suffix = sql.SQL(""" ) AS t (
            experiment_parameters,
            experiment_attributes,
            {experiment_columns},
            run_parameters,
            {run_columns}
            )
),
to_insert as (
    SELECT
        experiment_parameters,
        experiment_attributes,
        {experiment_columns}
    FROM query_values
    WHERE experiment_id IS NULL
),
inserted as (
    INSERT INTO {experiment_table} e (
        experiment_parameters,
        experiment_attributes,
        {experiment_columns}
    )
    SELECT *
    FROM to_insert
    ON CONFLICT (experiment_parameters) DO UPDATE
        SET experiment_id = e.experiment_id WHERE FALSE
    RETURNING 
        experiment_id, 
        experiment_parameters
),
retried AS (
    INSERT INTO {experiment_table} e (
        experiment_parameters,
        experiment_attributes,
        {experiment_columns}
    )
    SELECT to_insert.*
    FROM
        to_insert
        LEFT JOIN inserted USING (experiment_parameters)
    WHERE inserted.experiment_id IS NULL
    ON CONFLICT (experiment_parameters) DO UPDATE
        SET experiment_id = p.experiment_id WHERE FALSE
    RETURNING 
        experiment_id, 
        experiment_parameters
   )
),
INSERT INTO {run_table} (
    experiment_id,
    run_parameters,
    {run_columns}
    )
SELECT
    COALESCE (
        query_values.experiment_id,
        new_experiments.experiment_id
    ) experiment_id,
    run_parameters,
    {run_columns}
FROM
    query_values
    LEFT JOIN (
        SELECT * FROM inserted
        UNION ALL
        SELECT * FROM retried
        ) new_experiments USING (experiment_parameters)
ON CONFLICT DO NOTHING
RETURNING 
    experiment_id
;""").format(
            experiment_columns=experiment_columns_sql,
            run_columns=run_columns_sql,
            experiment_table=sql.Identifier(self._experiment_table),
            run_table=sql.Identifier(self._run_table),
        )

        # initialize parameter map
        with CursorManager(self._credentials) as cursor:
            self._parameter_map = PostgresParameterMap(cursor)

    def log(
        self,
        result: TaskResultRecord,
    ) -> None:
        with CursorManager(self._credentials) as cursor:

            def get_ids(parameter_dict: Dict[str, Any]) -> List[int]:
                return self._parameter_map.to_sorted_parameter_ids(
                    parameter_dict, cursor)

            def extract_values(target, columns):
                return [target.pop(c, None) for c in columns]

            experiment_parameters = get_ids(result.experiment_parameters)
            experiment_values = extract_values(result.experiment_data,
                                               self._experiment_columns)
            experiment_attributes = get_ids(result.experiment_data)

            run_values = extract_values(result.run_data, self._run_columns)
            run_parameters = get_ids(result.run_data)

            sql_values = sql.SQL(',').join(
                sql.Literal(v) for v in (
                    experiment_parameters,
                    experiment_attributes,
                    *experiment_values,
                    run_parameters,
                    *run_values,
                ))

            cursor.execute(self._log_query_prefix + sql_values +
                           self._log_query_suffix)

    def _make_column_sql(
        self,
        columns: List[Tuple[str, str]],
    ) -> Tuple[sql.Composed, sql.Composed]:
        columns_sql = sql.SQL(',').join(
            [sql.Identifier(x[0]) for x in columns])
        cast_columns_sql = sql.SQL(',').join([
            sql.SQL('{}::{}').format(sql.Identifier(x[0]), sql.SQL(x[1]))
            for x in columns
        ])
        return columns_sql, cast_columns_sql

    # @staticmethod
    # def _make_values_array(cursor, values: Iterable[Tuple]) -> sql.SQL:
    #     return sql.SQL(','.join(
    #         (cursor.mogrify('(' + (','.join(['%s' for _ in e])) + ')',
    #                         e).decode("utf-8") for e in values)))


'''
        + encode data payload as parquet bytestream
        + encode run indexing/table data as columns
        + canonicalize experiment parameters
        + get experiment id (and write new experiment if needed)
        + store it all


        + experiment parameters: (GIN index) -- use jsonb or ints?
            + all Task parameters, except for:
                + seed
                + batch
                + task_version
                + save_every_epochs
            + ml_task
            
        + experiment columns:
            ('num_free_parameters', 'bigint'),
            ('widths', 'integer[]'),
            ('network_structure', 'jsonb'),
            + input_shape
            + output_shape

        + run parameters:
            None

        + run columns:
            experiment_id
            run_id (needed? same as task id?)
            timestamp (needed? can check job time...)
            run data
            ('save_every_epochs', 'smallint'),
            ('platform', 'text'),
            ('git_hash', 'text'),
            ('hostname', 'text'),
            ('slurm_job_id', 'text'),
            ('num_gpus', 'integer'),
            ('num_nodes', 'integer'),
            ('num_cpus', 'integer'),
            ('gpu_memory', 'integer'),
            ('nodes', 'text'),
            ('cpus', 'text'),
            ('gpus', 'text'),
            ('strategy', 'text'),
            
        + run data:
            ('seed', 'bigint'),
            
            + history records:
                ('train_loss', 'real[]'),
                ('train_accuracy', 'real[]'),
                ('train_mean_squared_error', 'real[]'),
                ('train_mean_absolute_error', 'real[]'),
                ('train_root_mean_squared_error', 'real[]'),
                ('train_mean_squared_logarithmic_error', 'real[]'),
                ('train_hinge', 'real[]'),
                ('train_squared_hinge', 'real[]'),
                ('train_cosine_similarity', 'real[]'),
                ('train_kullback_leibler_divergence', 'real[]'),

                ('test_loss', 'real[]'),            
                ('test_accuracy', 'real[]'),            
                ('test_mean_squared_error', 'real[]'),
                ('test_mean_absolute_error', 'real[]'),
                ('test_root_mean_squared_error', 'real[]'),            
                ('test_mean_squared_logarithmic_error', 'real[]'),            
                ('test_hinge', 'real[]'),            
                ('test_squared_hinge', 'real[]'),            
                ('test_cosine_similarity', 'real[]'),            
                ('test_kullback_leibler_divergence', 'real[]'),

                ('validation_loss', 'real[]'),
                ('validation_accuracy', 'real[]'),
                ('validation_mean_squared_error', 'real[]'),
                ('validation_mean_absolute_error', 'real[]'),
                ('validation_root_mean_squared_error', 'real[]'),
                ('validation_mean_squared_logarithmic_error', 'real[]'),
                ('validation_hinge', 'real[]'),
                ('validation_squared_hinge', 'real[]'),
                ('validation_cosine_similarity', 'real[]'),
                ('validation_kullback_leibler_divergence', 'real[]'),

                ('parameter_count', 'bigint[]'),
                ('growth_points', 'smallint[]'),
        '''