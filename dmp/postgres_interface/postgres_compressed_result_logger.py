from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
import io
from numpy import place
from psycopg.sql import SQL, Composed, Identifier
from psycopg.types.json import Jsonb
from jobqueue.connection_manager import ConnectionManager
from dmp.logging.experiment_result_logger import ExperimentResultLogger
from dmp.postgres_interface.column_group import ColumnGroup
from dmp.postgres_interface.table_data import TableData
from dmp.postgres_interface.postgres_schema import PostgresSchema
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord

from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder


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
            experiment['properties'],
            experiment['values'],
        ))

        run_groups = ColumnGroup.concatenate((
            run['values'],
            run['data'],
            run['history'],
            run['extended_history'],
        ))

        values_groups = experiment_groups + run_groups
        self._values_groups = values_groups

        input_table = Identifier('_input')
        inserted_experiment_table = Identifier('_inserted')
        
        self._log_multiple_query_prefix = SQL("""
WITH {input_table} as (
    SELECT
        {casting_clause}
    FROM
        ( VALUES """).format(
            input_table=input_table,
            casting_clause=values_groups.casting_sql,
            values_placeholders=values_groups.placeholders,
            experiment_columns=experiment_groups.columns_sql,
            run_value_columns=run_groups.columns_sql,
            inserted_experiment_table=inserted_experiment_table,
            experiment_table=experiment.identifier,
            run_table=run.identifier,
            run_experiment_uid=schema.experiment_id_group.identifier,
        )

        self._log_multiple_query_suffix = SQL("""
) AS t (
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
            casting_clause=values_groups.casting_sql,
            values_placeholders=values_groups.placeholders,
            experiment_columns=experiment_groups.columns_sql,
            run_value_columns=run_groups.columns_sql,
            inserted_experiment_table=inserted_experiment_table,
            experiment_table=experiment.identifier,
            run_table=run.identifier,
            run_experiment_uid=schema.experiment_id_group.identifier,
        )

    def log(self,
            records: Union[Sequence[ExperimentResultRecord],
                           ExperimentResultRecord],
            connection=None) -> None:
        if connection is None:
            with ConnectionManager(self._schema.credentials) as connection:
                return self.log(records, connection)

        if isinstance(records, ExperimentResultRecord):
            return self.log((records, ), connection)

        if not isinstance(records, Sequence):
            raise ValueError(f'Invalid record type {type(records)}.')

        if len(records) <= 0:
            return

        run_data = []
        for record in records:
            experiment_column_values = self._schema.experiment[
                'values'].extract_column_values(record.experiment_attrs)
            
            experiment_column_values = self._schema.experiment[
                'values'].extract_column_values(record.experiment_properties)

            experiment_attrs = \
                self._schema.attribute_map.to_sorted_attr_ids(
                    record.experiment_attrs)
            experiment_properties = self._schema.attribute_map.to_sorted_attr_ids(
                    record.experiment_properties)

            experiment_uid = self._schema.make_experiment_uid(experiment_attrs)

            run_column_values = self._schema.run[
                'values'].extract_column_values(record.run_data)

            run_history = self._schema.make_history_bytes(record.run_history)
            run_extended_history = self._schema.make_history_bytes(
                record.run_extended_history)
            run_data.append((
                experiment_uid,
                experiment_attrs,
                experiment_properties,
                *experiment_column_values,
                *run_column_values,
                Jsonb(record.run_data),
                run_history,
                run_extended_history,
            ))

        placeholders = sql_comma.join(
            [SQL('({})').format(self._values_groups.placeholders)] *
            len(run_data))

        query = SQL("""{}{}{}""").format(
            self._log_multiple_query_prefix,
            placeholders,
            self._log_multiple_query_suffix,
        )

        connection.execute(
            query,
            list(chain(*run_data)),
            binary=True,
        )
