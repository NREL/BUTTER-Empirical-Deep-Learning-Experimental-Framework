from collections import Callable
from dataclasses import dataclass
import io
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type
from jobqueue.connect import load_credentials
from jobqueue.connection_manager import ConnectionManager

from jobqueue.job import Job
from psycopg.sql import SQL, Composed, Identifier, Literal
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.task import Task
from dmp.task.task_result import TaskResult
from dmp.worker import Worker
from dmp.postgres_interface.postgres_interface_common import sql_comma
from dmp.common import flatten, marshal_type_key

_summarizer_map: Dict[str, Callable[[Sequence[ExperimentResultRecord]],
                                    ExperimentSummaryRecord]] = {}


@dataclass
class UpdateExperimentSummary(Task):

    @staticmethod
    def register_types(types: Iterable[Type]) -> None:
        for target_type in types:
            UpdateExperimentSummary.register_type(target_type)

    @staticmethod
    def register_type(target_type: Type) -> None:
        type_code = target_type.__name__
        _summarizer_map[type_code] = target_type.summarize

    def __call__(self, worker: Worker, job: Job) -> TaskResult:
        experiment_limit = 64

        schema = worker._schema
        experiment = schema.experiment
        run = schema.run
        summary = schema.experiment_summary

        run_columns = ColumnGroup(
            run.experiment_id,
            run.run_timestamp,
            run.values,
            run.run_data,
            run.run_history,
        )

        experiment_summary_columns = ColumnGroup(
            summary.experiment_id,
            summary.last_run_timestamp,
            summary.data,
        )

        #                             AND experiment_properties @> array[2568] and experiment_attrs @> array[75, 224]
        lock_and_get_query = SQL("""
SELECT
    {selection}.{last_run_timestamp},
    {selection}.{experiment_attrs},
    {selection}.{experiment_properties},
    {run_columns}
FROM
    (
        SELECT DISTINCT ON({experiment_id})
            {selection}.{experiment_id},
            {selection}.{experiment_attrs},
            {selection}.{experiment_properties},
            MAX({run_timestamp}) OVER (PARTITION BY {experiment_id}) {last_run_timestamp}
        FROM
            (
                SELECT 
                    {run}.{run_schematimestamp}, 
                    {selection}.{experiment_id},
                    {selection}.{experiment_attrs},
                    {selection}.{experiment_properties}
                FROM 
                    {run} LEFT JOIN {experiment_summary_table} ON (
                        {run}.{experiment_id} = {experiment_summary_table}.{experiment_id}
                        AND {experiment_summary_table}.{last_run_timestamp} > {run_timestamp})
                    CROSS JOIN LATERAL (
                        SELECT
                            {experiment}.{experiment_id},
                            {experiment}.{experiment_attrs},
                            {experiment}.{experiment_properties}
                        FROM
                            {experiment}
                        WHERE
                            {experiment}.{experiment_id} = {run}.{experiment_id}
                        FOR UPDATE SKIP LOCKED
                    ) {selection}
                WHERE 
                    {run_timestamp} > (
                        SELECT COALESCE(MAX({experiment_summary_table}.{run_update_limit}), '1960-01-01'::timestamp)
                        FROM {experiment_summary_table}
                        )
                    AND {experiment_summary_table}.{experiment_id} IS NULL
                ORDER BY {run}.{run_timestamp} ASC
                LIMIT {experiment_limit}
            ) {selection}
    ) {selection}
    CROSS JOIN LATERAL (
        SELECT
            {run_columns}
        FROM 
            {run} 
        WHERE 
            {run}.{run_experiment_id} = {selection}.{experiment_id}
    ) {run}
;""").format(
            selection=Identifier('_selection'),
            last_run_timestamp=summary.last_run_timestamp.identifier,
            experiment_attrs=experiment.experiment_attrs.identifier,
            experiment_properties=experiment.experiment_properties.identifier,
            run_columns=run_columns.of(run.identifier),
            experiment_id=experiment.experiment_id.identifier,
            run_experiment_id=run.experiment_id.identifier,
            run_timestamp=run.run_timestamp.identifier,
            run=run.identifier,
            experiment_summary_table=summary.identifier,
            experiment=experiment.identifier,
            run_update_limit=summary.run_update_limit.identifier,
            experiment_limit=Literal(experiment_limit),
        )

        def make_update_progress_query(num_summaries: int) -> Composed:
            return SQL("""
INSERT INTO {experiment_summary_table}
(
    {summary_columns}
)
VALUES {summary_column_placeholders}
ON CONFLICT ({experiment_id}) DO UPDATE SET
    {update_clause}
;""").format(
                experiment_summary_table=summary.identifier,
                summary_columns=experiment_summary_columns.identifiers,
                summary_column_placeholders=sql_comma.join([
                    SQL('({})').format(experiment_summary_columns.placeholders)
                ] * num_summaries),
                experiment_id=experiment_id,
                update_clause=sql_comma.join(
                    (SQL('{c}=EXCLUDED.{c}').format(c=c)
                     for c in experiment_summary_columns.identifiers)),
            )

        result_columns = ColumnGroup(
            summary.last_run_timestamp,
            experiment.experiment_attrs,
            experiment.experiment_properties,
            run_columns,
        )

        with ConnectionManager(schema.credentials) as connection:
            with connection.transaction():
                with connection.cursor(binary=True) as cursor:

                    # lock experiments and get runs to summarize
                    summary_rows = []
                    experiment_id = None
                    runs = []
                    last_updated = None
                    experiment_attrs = {}
                    experiment_properties = {}

                    cursor.execute(lock_and_get_query, binary=True)
                    for row in cursor.fetchall():

                        def value_of(column: Column) -> Any:
                            return row[result_columns[column]]

                        row_uid = value_of(run.experiment_id)
                        if row_uid != experiment_id:
                            if len(runs) > 0:
                                summary_rows.append(
                                    (last_updated,
                                     self._compute_summary(runs)))

                            experiment_id = row_uid
                            runs.clear()
                            experiment_attrs = schema.attribute_map.attribute_map_from_ids(
                                value_of(experiment.experiment_attrs))
                            experiment_properties = schema.attribute_map.attribute_map_from_ids(
                                value_of(experiment.experiment_properties))
                            last_updated = value_of(run.run_timestamp)

                        last_updated = max(
                            last_updated,  # type: ignore
                            value_of(run.run_timestamp),
                        )
                        run_data = value_of(run.run_data)
                        for c in run.values:
                            run_data[c] = value_of(c)

                        run_history = schema.convert_bytes_to_dataframe(
                            value_of(run.run_history))

                        runs.append(
                            ExperimentResultRecord(
                                experiment_attrs,
                                experiment_properties,
                                run_data,
                                run_history,  # type: ignore
                                None,
                            ))

                    if len(runs) > 0:
                        summary_rows.append((
                            experiment_id,
                            last_updated,
                            *self._summary_to_bytes(
                                schema, self._compute_summary(runs)),
                        ))

                    # num_summaries = len(summaries)
                    # if num_summaries > 0:
                    #     # write summaries to database
                    #     cursor.execute(
                    #         make_update_progress_query(num_summaries),
                    #         list(
                    #             flatten(
                    #                 ((
                    #                     last_updated,
                    #                     summary.experiment_id,
                    #                     summary.core_data,
                    #                     summary.extended_data,
                    #                 ) for last_updated, summary in summaries),
                    #                 levels=1,
                    #             )),
                    #     )

        return TaskResult()

    def _compute_summary(self, runs: List[ExperimentResultRecord]):
        experiment_type_code = runs[0].experiment_attrs[marshal_type_key]
        return _summarizer_map[experiment_type_code](runs)

    def _summary_to_bytes(
        self,
        schema: PostgresSchema,
        summary: ExperimentSummaryRecord,
    ) -> Sequence[Optional[bytes]]:
        return [
            schema.convert_dataframe_to_bytes(df) for df in (
                summary.by_epoch,
                summary.by_loss,
                summary.by_progress,
                summary.epoch_subset,
            )
        ]
