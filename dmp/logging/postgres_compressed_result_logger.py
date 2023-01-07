
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

psycopg2.extras.register_default_json(loads=simplejson.loads, globally=True)  # type: ignore
psycopg2.extras.register_default_jsonb(loads=simplejson.loads, globally=True)  # type: ignore
psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)  # type: ignore

class PostgresCompressedResultLogger(ResultLogger):
    _credentials: Dict[str, Any]
    _run_columns: List[Tuple[str, str]]
    _log_query_prefix: sql.Composed
    _log_query_suffix: sql.Composed
    _parameter_map: PostgresParameterMap

    def __init__(self,
                 credentials: Dict[str, Any],
                 ) -> None:
        super().__init__()
        self._credentials = credentials

        self.run_table:str= 'run'
        self.experiment_table : str = 'experiment'

        self._experiment_columns = sorted([
            ('network_structure', 'jsonb'),
        ])

        self._run_columns = sorted([
            ('job_id', 'uuid'),
            ('run_id', 'uuid'),
            ('slurm_job_id', 'bigint'),
            ('task_version', 'smallint'),
            ('num_gpus', 'smallint'),
            ('num_cpus', 'smallint'),
            ('num_nodes', 'smallint'),
            ('gpu_memory', 'integer'),
            ('host_name', 'text'),
            ('batch', 'text'),
        ])

        experiment_columns_sql, cast_experiment_columns_sql = \
            self._make_column_sql(self._experiment_columns)
        run_columns_sql, cast_run_columns_sql = \
            self._make_column_sql(self._run_columns)

        self._log_query_prefix = sql.SQL("""
WITH v as (
    SELECT
        experiment_parameters::smallint[] experiment_parameters,
        experiment_data::jsonb
        run_parameters::smallint[] run_parameters,
        {cast_experiment_columns},
        {cast_run_columns}
    FROM
        (VALUES """).format(
            cast_experiment_columns=cast_experiment_columns_sql,
            cast_run_columns=cast_run_columns_sql,
        )

        self._log_query_suffix = sql.SQL(""" ) AS t (
            job_id,
            run_id,
            experiment_parameters,
            run_parameters,
            {experiment_columns},
            {run_columns}
            )
),
exp_to_insert as (
    SELECT * FROM
    (
        SELECT distinct experiment_parameters
        FROM v
        WHERE NOT EXISTS (SELECT * from {experiment_table} ex where ex.experiment_parameters = v.experiment_parameters)
    ) d,
    lateral (
        SELECT {experiment_columns}
        FROM v
        WHERE v.experiment_parameters = d.experiment_parameters
        LIMIT 1
    ) l
),
ir as (
    INSERT INTO {experiment_table} (
        experiment_parameters,
        {experiment_columns}
    )
    SELECT * FROM exp_to_insert
    ON CONFLICT DO NOTHING
    RETURNING 
        experiment_id, 
        experiment_parameters
),
i as (
    SELECT * FROM ir
    UNION ALL
    SELECT 
        e.experiment_id, 
        v.experiment_parameters
    FROM
        v,
        {experiment_table} e
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
)
INSERT INTO {run_table} (
    experiment_id,
    run_id,
    job_id,
    run_parameters,
    {run_columns}
    )
SELECT
    experiment_id,
    run_id,
    job_id,
    run_parameters,
    {run_columns}
FROM
    x
ON CONFLICT DO NOTHING
;
        """).format(
            experiment_columns=experiment_columns_sql,
            run_columns=run_columns_sql,
            experiment_table=sql.Identifier(experiment_table),
            run_table=sql.Identifier(run_table),
        )

        # initialize parameter map
        with CursorManager(self._credentials) as cursor:
            self._parameter_map = PostgresParameterMap(cursor)
        pass

    def log(
        self,
        result: TaskResultRecord,
        job_id: UUID,
        worker_id: UUID,
    ) -> None:
        # TODO: finish compressed logger
        # TODO: does it make sense to canonicalize columns?
        
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
        


        
        

        with CursorManager(self._credentials) as cursor:
            log_entries = [
                self._transform_row(cursor, *result)
                for result in results]
            values = self._make_values_array(cursor, log_entries)
            # print(cursor.mogrify(self._log_query_prefix + values + self._log_query_suffix))
            cursor.execute(
                self._log_query_prefix + values + self._log_query_suffix)

    def _transform_row(
        self,
        cursor,
        job_id: UUID,
        run_id: UUID,
        result: Dict[str, Any],
    ):
        def extract_values(columns):
            return [result.pop(c[0], None) for c in columns]

        experiment_column_values = extract_values(self._experiment_columns)
        run_column_values = extract_values(self._run_columns)

        experiment_parameters = result.copy()
        for k in self._run_only_parameters:
            experiment_parameters.pop(k, None)
        run_parameters = result

        # get sorted parameter ids
        run_parameter_ids = \
            self._parameter_map.to_sorted_parameter_ids(
                run_parameters, cursor)

        experiment_parameter_ids = \
            self._parameter_map.to_sorted_parameter_ids(
                experiment_parameters, cursor)

        return (
            job_id,
            run_id,
            experiment_parameter_ids,
            run_parameter_ids,
            *experiment_column_values,
            *run_column_values,
        )

    def _make_column_sql(
        self,
        columns: List[Tuple[str, str]],
    ) -> Tuple[sql.Composed, sql.Composed]:
        columns_sql = sql.SQL(',').join(
            [sql.Identifier(x[0]) for x in columns])
        cast_columns_sql = sql.SQL(',').join(
            [sql.SQL('{}::{}').format(
                sql.Identifier(x[0]), sql.SQL(x[1]))
                for x in columns])
        return columns_sql, cast_columns_sql

    @staticmethod
    def _make_values_array(cursor, values: Iterable[Tuple]) -> sql.SQL:
        return sql.SQL(','.join((
            cursor.mogrify(
                '(' + (','.join(['%s' for _ in e])) + ')', e).decode("utf-8")
            for e in values)))
