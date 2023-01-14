from typing import Any, Dict, Iterable, Optional, Tuple, List
import io
import uuid
import hashlib
from jobqueue.connection_manager import ConnectionManager
from psycopg import sql
import psycopg
import pyarrow
import pyarrow.parquet
from dmp.logging.postgres_attribute_map import PostgresAttributeMap
from dmp.logging.result_logger import ResultLogger

from dmp.parquet_util import make_pyarrow_schema

from dmp.task.task_result_record import TaskResultRecord

# psycopg.extras.register_uuid()
# psycopg.extras.register_default_json(loads=simplejson.loads,
#                                       globally=True)  # type: ignore
# psycopg.extras.register_default_jsonb(loads=simplejson.loads,
#                                        globally=True)  # type: ignore
# psycopg.extensions.register_adapter(dict,
#                                      psycopg.extras.Json)  # type: ignore


class PostgresCompressedResultLogger(ResultLogger):
    _credentials: Dict[str, Any]
    _run_table: sql.Identifier
    _experiment_table: sql.Identifier

    _experiment_columns: List[Tuple[str, str]] = [
        ('experiment_id', 'integer'),
    ]

    _run_columns: List[Tuple[str, str]] = [
        ('run_id', 'uuid'),
        ('job_id', 'uuid'),
        ('seed', 'bigint'),
        ('slurm_job_id', 'bigint'),
        ('task_version', 'smallint'),
        ('num_nodes', 'smallint'),
        ('num_cpus', 'smallint'),
        ('num_gpus', 'smallint'),
        ('gpu_memory', 'integer'),
        ('host_name', 'text'),
        ('batch', 'text'),
        ('run_data', 'jsonb'),
        ('run_history', 'bytea'),
    ]

    _run_data_column_index: int = -2
    _run_history_column_index: int = -1

    _log_query_prefix: sql.Composed
    _log_query_suffix: sql.Composed
    _parameter_map: PostgresAttributeMap

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

        values_placeholders = sql.SQL(',').join((sql.SQL('%b') for i in range(2 + len(self._experiment_columns) + len(self._run_columns))))
        self._log_query_prefix = sql.SQL("""
        WITH query_values as (
            SELECT
                experiment_uid::uuid,
                experiment_attributes::integer[] experiment_attributes,
                {cast_experiment_columns},
                {cast_run_columns}
            FROM
                ( VALUES ({values_placeholders}) ) AS t (
                    experiment_uid,
                    experiment_attributes,
                    {experiment_columns},
                    {run_columns}
                    )
        ),
        inserted_experiment as (
            INSERT INTO {experiment_table} AS e (
                experiment_uid,
                experiment_attributes,
                {experiment_columns}
            )
            SELECT
                experiment_uid,
                experiment_attributes,
                {experiment_columns}
            FROM query_values
            ON CONFLICT DO NOTHING
        )
        INSERT INTO {run_table} (
            experiment_uid,
            {run_columns}
            )
        SELECT 
            experiment_uid,
            {run_columns}
        FROM query_values
        ON CONFLICT DO NOTHING
        ;""").format(
                    experiment_table=self._experiment_table,
                    cast_experiment_columns=cast_experiment_columns,
                    cast_run_columns=cast_run_columns,
                    values_placeholders=values_placeholders,
                    experiment_columns=experiment_columns,
                    run_columns=run_columns,
                    run_table=self._run_table,
                )
#         self._log_query_prefix = sql.SQL("""
# WITH query_values as (
#     SELECT
#         experiment_uid::uuid,
#         experiment_attributes::integer[] experiment_attributes,
#         {cast_experiment_columns},
#         {cast_run_columns}
#     FROM
#         ( VALUES ( """).format(
#             experiment_table=self._experiment_table,
#             cast_experiment_columns=cast_experiment_columns,
#             cast_run_columns=cast_run_columns,
#         )

#         self._log_query_suffix = sql.SQL(""" ) ) AS t (
#             experiment_uid,
#             experiment_attributes,
#             {experiment_columns},
#             {run_columns}
#             )
# ),
# inserted_experiment as (
#     INSERT INTO {experiment_table} AS e (
#         experiment_uid,
#         experiment_attributes,
#         {experiment_columns}
#     )
#     SELECT
#         experiment_uid,
#         experiment_attributes,
#         {experiment_columns}
#     FROM query_values
#     ON CONFLICT DO NOTHING
# )
# INSERT INTO {run_table} (
#     experiment_uid,
#     {run_columns}
#     )
# SELECT 
#     experiment_uid,
#     {run_columns}
# FROM query_values
# ON CONFLICT DO NOTHING
# ;""").format(
#             experiment_columns=experiment_columns,
#             run_columns=run_columns,
#             experiment_table=self._experiment_table,
#             run_table=self._run_table,
#         )

        # initialize parameter map
        with ConnectionManager(self._credentials) as connection:
            self._parameter_map = PostgresAttributeMap(connection)

    @staticmethod
    def make_experiment_uid(experiment_parameters):
        uid_string = '{' + ','.join(str(i)
                                    for i in experiment_parameters) + '}'
        return uuid.UUID(hashlib.md5(uid_string.encode('utf-8')).hexdigest())

    def log(
        self,
        result: TaskResultRecord,
        connection = None
    ) -> None:
        if connection is None:
            with ConnectionManager(self._credentials) as connection:
                self.log(result, connection)
            return

        experiment_parameters = self._get_ids(result.experiment_parameters,
                                                connection)
        experiment_uid = self.make_experiment_uid(experiment_parameters)

        experiment_column_values = PostgresCompressedResultLogger._extract_values(
            result.experiment_data,
            self._experiment_columns,
        )

        experiment_attributes = self._get_ids(result.experiment_data,
                                                connection)
        experiment_attributes.extend(experiment_parameters)
        experiment_attributes.sort()

        run_column_values = PostgresCompressedResultLogger._extract_values(
            result.run_data, self._run_columns)
        run_column_values[self._run_data_column_index] = \
            psycopg.types.json.Jsonb(result.run_data)

        # print(f'run_column_values {run_column_values}\nrun_attributes {run_attributes}')

        with io.BytesIO() as history_buffer:
            self._make_history_bytes(
                result.run_history,  # type: ignore
                history_buffer,
            )
            run_column_values[self._run_history_column_index] = \
                history_buffer.getvalue()

        # sql_values = sql.SQL(', ').join(
        #     sql.Literal(v) for v in (
        #         experiment_uid,
        #         experiment_attributes,
        #         *experiment_column_values,
        #         *run_column_values,
        #     ))

        # query = self._log_query_prefix + sql_values + self._log_query_suffix
        connection.execute(self._log_query_prefix, (
                experiment_uid,
                experiment_attributes,
                *experiment_column_values,
                *run_column_values,
            ), binary=True)
        # with psycopg.ClientCursor(connection) as cursor:
        #     print(cursor.mogrify(query))
        #     cursor.execute(query)

    # def _split_data_fields(
    #     self,
    #     cursor,
    #     all_data: Dict[str, Any],
    #     data_columns: Iterable[Tuple[str, str]],
    #     data_keys: Optional[Iterable[str]],
    #     data_index: int,
    # ) -> Tuple[List[Any], List[int]]:
    #     column_values = PostgresCompressedResultLogger._extract_values(
    #         all_data, data_columns)
    #     data_map = PostgresCompressedResultLogger._extract_map(
    #         all_data, data_keys)
    #     column_values[data_index] = data_map
    #     data_attributes = self._get_ids(all_data, cursor)
    #     return column_values, data_attributes

    @staticmethod
    def _extract_values(
        target: Dict[str, Any],
        columns: Iterable[Tuple[str, str]],
    ) -> List[Any]:
        result = []
        for name, type_name in columns:
            value = target.pop(name, None)
            if type_name == 'jsonb' and value is not None:
                value = psycopg.types.json.Jsonb(value)
            result.append(value)
        return result

    @staticmethod
    def _extract_map(
        target: Dict[str, Any],
        columns: Optional[Iterable[str]],
    ) -> Optional[Dict[str, Any]]:
        if columns is None:
            return None
        return psycopg.types.json.Jsonb(
            {name: target.pop(name, None)
             for name in columns})

    def _get_ids(
        self,
        parameter_dict: Dict[str, Any],
        connection,
    ) -> List[int]:
        return self._parameter_map.to_sorted_attribute_ids(
            parameter_dict,
            connection=connection,
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