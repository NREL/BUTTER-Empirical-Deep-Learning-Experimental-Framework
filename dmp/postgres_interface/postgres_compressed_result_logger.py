from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
import io
from numpy import place
from psycopg.sql import SQL, Composed, Identifier
from psycopg.types.json import Jsonb
from jobqueue.connection_manager import ConnectionManager
from dmp.logging.experiment_result_logger import ExperimentResultLogger
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
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

        experiment = self._schema.experiment
        run = self._schema.run
        experiment_all = experiment.all
        insertion_columns = run.insertion_columns
        self._values_columns = experiment_all + insertion_columns
        input_table = Identifier('_input')

        self._log_multiple_query_prefix = SQL("""
WITH {input_table} as (
    SELECT
        {casting_clause}
    FROM
        ( VALUES """).format(
            input_table=input_table,
            casting_clause=self._values_columns.casting_sql,
        )

        self._log_multiple_query_suffix = SQL("""
) AS t (
            {experiment_columns},
            {run_value_columns}
            )
),
{_inserted} as (
    INSERT INTO {experiment} (
        {experiment_columns}
    )
    SELECT
        {experiment_columns}
    FROM 
        {input_table}
    WHERE
        NOT EXISTS (
            SELECT 1
            FROM {experiment} e
            WHERE 
                e.{experiment_id} = {input_table}.{experiment_id}
                AND e.{experiment_tags} @> {input_table}.{experiment_tags}
        )
    ON CONFLICT ({experiment_id}) DO UPDATE SET
        {experiment_tags} = (SELECT array_agg(attr_id) FROM (
            SELECT attr_id
            FROM (
                SELECT 
                    attr_id
                FROM 
                    unnest({experiment}.{experiment_tags}) a(attr_id)
                UNION ALL
                SELECT 
                    attr_id
                FROM 
                    unnest(EXCLUDED.{experiment_tags}) a(attr_id)
                ) _tmp
            GROUP BY attr_id
            ORDER BY attr_id ASC
            ) _tmp )
)
INSERT INTO {run} (
    {experiment_id},
    {run_value_columns}
    )
SELECT 
    {experiment_id},
    {run_value_columns}
FROM {input_table}
ON CONFLICT DO NOTHING
;""").format(
            experiment_tags=experiment.experiment_tags.identifier,
            experiment_columns=experiment_all.columns_sql,
            run_value_columns=insertion_columns.columns_sql,
            _inserted=Identifier('_inserted'),
            experiment=experiment.identifier,
            run=run.identifier,
            experiment_id=experiment.experiment_id.identifier,
            input_table=input_table,
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

        schema = self._schema
        attribute_map = schema.attribute_map
        value_columns = schema.experiment.values
        run_value_columns = schema.run.values
        run_values = []
        for record in records:
            experiment_column_values = value_columns.extract_column_values(
                record.experiment_attrs,
                record.experiment_tags,
            )

            experiment_attrs = attribute_map.to_sorted_attr_ids(
                record.experiment_attrs)

            run_values.append([
                schema.make_experiment_id(experiment_attrs),
                experiment_attrs,
                attribute_map.to_sorted_attr_ids(record.experiment_tags),
                *experiment_column_values,
                *run_value_columns.extract_column_values(record.run_data),
                Jsonb(record.run_data),
                schema.convert_dataframe_to_bytes(record.run_history),
                schema.convert_dataframe_to_bytes(record.run_extended_history),
            ])
            print(record.run_data)

        placeholders = sql_comma.join(
            [SQL('({})').format(self._values_columns.placeholders)] *
            len(run_values))

        query = SQL("""{}{}{}""").format(
            self._log_multiple_query_prefix,
            placeholders,
            self._log_multiple_query_suffix,
        )

        
        print(query)
        run_payload = tuple(chain(*run_values))
        print(run_payload)

        connection.execute(
            query,
            run_payload,
            binary=True,
        )
