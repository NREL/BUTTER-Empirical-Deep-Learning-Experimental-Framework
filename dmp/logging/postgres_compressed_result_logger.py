from uuid import UUID
from typing import Any, Dict, Iterable, Optional, Tuple, List
import io
from psycopg2 import sql
import simplejson
import psycopg2
import pyarrow
from dmp.logging.postgres_parameter_map import PostgresParameterMap
from dmp.logging.result_logger import ResultLogger

from jobqueue.cursor_manager import CursorManager

from dmp.parquet_util import make_pyarrow_schema

from dmp.task.task_result_record import TaskResultRecord

psycopg2.extras.register_uuid()
psycopg2.extras.register_default_json(loads=simplejson.loads,
                                      globally=True)  # type: ignore
psycopg2.extras.register_default_jsonb(loads=simplejson.loads,
                                       globally=True)  # type: ignore
psycopg2.extensions.register_adapter(dict,
                                     psycopg2.extras.Json)  # type: ignore


class PostgresCompressedResultLogger(ResultLogger):
    _credentials: Dict[str, Any]
    _run_table: sql.Identifier
    _experiment_table: sql.Identifier

    _experiment_columns: List[Tuple[str, str]] = [
        ('experiment_data', 'jsonb'),
        ('model_structure', 'bytea'),
    ]

    _experiment_data_idx: int = 0
    _experiment_data_columns: Optional[List[str]] = None

    _run_columns: List[Tuple[str, str]] = [
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
        ('run_data', 'jsonb'),
        ('run_history', 'bytea'),
    ]

    _run_data_column_index: int = -2
    _run_history_column_index: int = -1

    _run_data_columns: Optional[List[str]] = None

    _log_query_prefix: sql.Composed
    _log_query_suffix: sql.Composed
    _insert_experiment_prefix: sql.Composed
    _insert_experiment_suffix: sql.Composed
    _parameter_map: PostgresParameterMap

    def __init__(
        self,
        credentials: Dict[str, Any],
    ) -> None:
        super().__init__()
        self._credentials = credentials

        self._run_table = sql.Identifier('run2')
        self._experiment_table = sql.Identifier('experiment2')

        experiment_columns, cast_experiment_columns = \
            self._make_column_sql(self._experiment_columns)
        run_columns, cast_run_columns = \
            self._make_column_sql(self._run_columns)

        self._log_query_prefix = sql.SQL("""
WITH query_values as (
    SELECT
        (   SELECT experiment_id 
            FROM {experiment_table} e 
            WHERE e.experiment_parameters = t.experiment_parameters::integer[]
            LIMIT 1
        ) experiment_id,
        experiment_parameters::integer[] experiment_parameters,
        experiment_attributes::integer[] experiment_attributes,
        {cast_experiment_columns},
        run_attributes::integer[] run_attributes,
        {cast_run_columns}
    FROM
        ( VALUES ( """).format(
            experiment_table=self._experiment_table,
            cast_experiment_columns=cast_experiment_columns,
            cast_run_columns=cast_run_columns,
        )

        self._log_query_suffix = sql.SQL(""" ) ) AS t (
            experiment_parameters,
            experiment_attributes,
            {experiment_columns},
            run_attributes,
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
    INSERT INTO {experiment_table} AS e (
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
    INSERT INTO {experiment_table} AS e (
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
        SET experiment_id = e.experiment_id WHERE FALSE
    RETURNING 
        experiment_id, 
        experiment_parameters
)
INSERT INTO {run_table} (
    experiment_id,
    run_attributes,
    {run_columns}
    )
SELECT
    COALESCE (
        query_values.experiment_id,
        new_experiments.experiment_id
    ) experiment_id,
    run_attributes,
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
            experiment_columns=experiment_columns,
            run_columns=run_columns,
            experiment_table=self._experiment_table,
            run_table=self._run_table,
        )

        self._insert_experiment_prefix = sql.SQL("""
INSERT INTO {experiment_table} AS e (
    experiment_id,
    experiment_parameters,
    experiment_attributes,
    {experiment_columns}
)
SELECT
    experiment_id::integer,
    experiment_parameters::integer[] experiment_parameters,
    experiment_attributes::integer[] experiment_attributes,
    {cast_experiment_columns}
FROM
    (VALUES ( 
""").format(
            experiment_table=self._experiment_table,
            experiment_columns=experiment_columns,
            cast_experiment_columns=cast_experiment_columns,
        )

        self._insert_experiment_suffix = sql.SQL(""" ) 
        ) AS t (
            experiment_id,
            experiment_parameters,
            experiment_attributes,
            {experiment_columns}
            )
ON CONFLICT DO NOTHING
""").format(experiment_columns=experiment_columns, )

        # initialize parameter map
        with CursorManager(self._credentials) as cursor:
            self._parameter_map = PostgresParameterMap(cursor)

    def log(
        self,
        result: TaskResultRecord,
        experiment_id: Optional[int] = None,
    ) -> None:
        with CursorManager(self._credentials) as cursor:

            experiment_parameters = self._get_ids(
                cursor,
                result.experiment_parameters,
            )

            experiment_column_data, experiment_attributes = self._split_data_fields(
                cursor,
                result.experiment_data,
                self._experiment_columns,
                self._experiment_data_columns,
                self._experiment_data_idx,
            )

            run_column_values, run_attributes = self._split_data_fields(
                cursor,
                result.run_data,
                self._run_columns,
                self._run_data_columns,
                self._run_data_column_index,
            )

            with io.BytesIO() as history_buffer:
                self._make_history_bytes(
                    result.run_history,  # type: ignore
                    history_buffer,
                )
                run_column_values[self._run_history_column_index] = \
                    history_buffer.getvalue()

            if experiment_id is not None:
                q = self._insert_experiment_prefix + (sql.SQL(', ').join(
                    sql.Literal(v) for v in (
                        experiment_id,
                        experiment_parameters,
                        experiment_attributes,
                        *experiment_column_data,
                    ))) + self._insert_experiment_suffix
                # print(cursor.mogrify(q).decode('utf-8'))
                cursor.execute(q)

            sql_values = sql.SQL(', ').join(
                sql.Literal(v) for v in (
                    experiment_parameters,
                    experiment_attributes,
                    *experiment_column_data,
                    run_attributes,
                    *run_column_values,
                ))

            query = self._log_query_prefix + sql_values + self._log_query_suffix
            # print(cursor.mogrify(query).decode('utf-8'))
            cursor.execute(query)

    def _split_data_fields(
        self,
        cursor,
        all_data: Dict[str, Any],
        data_columns: Iterable[Tuple[str, str]],
        data_keys: Optional[Iterable[str]],
        data_index: int,
    ) -> Tuple[List[Any], List[int]]:
        column_values = PostgresCompressedResultLogger._extract_values(
            all_data, data_columns)
        data_map = PostgresCompressedResultLogger._extract_map(
            all_data, data_keys)
        column_values[data_index] = data_map
        data_attributes = self._get_ids(all_data, cursor)
        return column_values, data_attributes

    @staticmethod
    def _extract_values(
        target: Dict[str, Any],
        columns: Iterable[Tuple[str, str]],
    ) -> List[Any]:
        return [target.pop(name, None) for name, type_name in columns]

    @staticmethod
    def _extract_map(
        target: Dict[str, Any],
        columns: Optional[Iterable[str]],
    ) -> Optional[Dict[str, Any]]:
        if columns is None:
            return None
        return {name: target.pop(name, None) for name in columns}

    def _get_ids(
        self,
        cursor,
        parameter_dict: Dict[str, Any],
    ) -> List[int]:
        return self._parameter_map.to_sorted_parameter_ids(
            parameter_dict,
            cursor,
        )

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

    def _make_history_bytes(
        self,
        history: dict,
        buffer: io.BytesIO,
    ) -> None:
        schema, use_byte_stream_split = make_pyarrow_schema(history.items())

        table = pyarrow.Table.from_pydict(history, schema=schema)

        pyarrow_file = pyarrow.PythonFile(buffer)

        pyarrow.parquet.write_table(
            table,
            pyarrow_file,
            # root_path=dataset_path,
            # schema=schema,
            # partition_cols=partition_cols,
            data_page_size=8 * 1024,
            # compression='BROTLI',
            # compression_level=8,
            compression='ZSTD',
            compression_level=12,
            use_dictionary=False,
            use_byte_stream_split=use_byte_stream_split,
            version='2.6',
            data_page_version='2.0',
            # existing_data_behavior='overwrite_or_ignore',
            # use_legacy_dataset=False,
            write_statistics=False,
            # write_batch_size=64,
            # dictionary_pagesize_limit=64*1024,
        )
        # bytes_written = buffer.getbuffer().nbytes

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