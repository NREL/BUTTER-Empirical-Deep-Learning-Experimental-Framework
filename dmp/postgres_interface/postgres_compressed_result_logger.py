from typing import Any, Dict, Iterable, Optional, Tuple, List
import io
from psycopg.sql import SQL, Composed, Identifier
from psycopg.types.json import Jsonb
from jobqueue.connection_manager import ConnectionManager
from dmp.logging.experiment_result_logger import ExperimentResultLogger
from dmp.postgres_interface.column_group import ColumnGroup
from dmp.postgres_interface.table_data import TableData
from dmp.postgres_interface.postgres_schema import PostgresSchema
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord

class PostgresCompressedResultLogger(ExperimentResultLogger):
    _schema: PostgresSchema
    _log_result_record_query: Composed

    def __init__(
        self,
        schema: PostgresSchema,
    ) -> None:
        super().__init__()
        self._schema = schema

        experiment: TableData = self._schema.experiment
        run: TableData = self._schema.run

        experiment_groups = ColumnGroup.concatenate((
            experiment['uid'],
            experiment['attrs'],
            experiment['values'],
        ))

        run_groups = ColumnGroup.concatenate((
            run['values'],
            run['data'],
            run['history'],
        ))

        values_groups = experiment_groups + run_groups

        input_table = Identifier('_input')
        inserted_experiment_table = Identifier('_inserted')
        self._log_result_record_query = SQL("""
WITH {input_table} as (
    SELECT
        {casting_clause}
    FROM
        ( VALUES ({values_placeholders}) ) AS t (
            {experiment_columns},
            {run_value_columns}
            )
),
{inserted_experiment_table} as (
    INSERT INTO {experiment_table} AS e (
        {experiment_columns}
    )
    SELECT
        {experiment_columns}
    FROM {input_table}
    ON CONFLICT DO NOTHING
)
INSERT INTO {run_table} (
    {run_experiment_uid},
    {run_value_columns}
    )
SELECT 
    {run_experiment_uid},
    {run_value_columns}
FROM {input_table}
ON CONFLICT DO NOTHING
;""").format(
            input_table=input_table,
            casting_clause=experiment_groups.casting_sql,
            values_placeholders=values_groups.placeholders,
            experiment_columns=experiment_groups.columns_sql,
            run_value_columns=run_groups.columns_sql,
            inserted_experiment_table=inserted_experiment_table,
            experiment_table=experiment.name_sql,
            run_table=run.name_sql,
            run_experiment_uid=run['experiment_uid'].columns_sql,
        )

    def log(self, record: ExperimentResultRecord, connection=None) -> None:
        if connection is None:
            with ConnectionManager(self._schema.credentials) as connection:
                self.log(record, connection)
            return

        experiment_column_values = self._schema.experiment[
            'value'].extract_column_values(record.experiment_attrs)

        experiment_attrs = \
            self._schema.attribute_map.to_sorted_attr_ids(
                record.experiment_attrs)

        experiment_uid = self._schema.make_experiment_uid(experiment_attrs)

        run_column_values = self._schema.run['value'].extract_column_values(
            record.run_data)

        run_history_bytes = None
        with io.BytesIO() as history_buffer:
            self._schema.make_history_bytes(
                record.run_history,  # type: ignore
                history_buffer,
            )
            run_history_bytes = history_buffer.getvalue()

        connection.execute(
            self._schema.log_result_record_query,
            (
                experiment_uid,
                experiment_attrs,
                *experiment_column_values,
                *run_column_values,
                Jsonb(record.run_data),
                run_history_bytes,
            ),
            binary=True,
        )
