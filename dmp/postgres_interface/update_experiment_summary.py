from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type
from jobqueue.connection_manager import ConnectionManager

from jobqueue.job import Job
from psycopg import ClientCursor
from psycopg.sql import SQL, Composed, Identifier, Literal
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.postgres_interface.update_experiment_summary_result import UpdateExperimentSummaryResult
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
        num_summaries = 0
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

        summary_columns = ColumnGroup(
            summary.last_run_timestamp,
            summary.experiment_id,
            summary.most_recent_run,
            summary.by_epoch,
            summary.by_loss,
            summary.by_progress,
            summary.epoch_subset,
        )

        selection_columns = ColumnGroup(
            experiment.experiment_id,
            experiment.experiment_attrs,
            experiment.experiment_properties,
        )

        #                             AND experiment_properties @> array[2568] and experiment_attrs @> array[75, 224]
        selection_table = Identifier('_selection')
        lock_and_get_query = SQL("""
SELECT
    {selection}.{last_run_timestamp},
    {selection_columns},
    {run_columns}
FROM
    (
        SELECT DISTINCT ON({experiment_id})
            {selection_columns},
            MAX({run_timestamp}) OVER (PARTITION BY {experiment_id}) {last_run_timestamp}
        FROM
            (
                SELECT 
                    {selection_columns},
                    {run}.{run_timestamp}
                FROM 
                    (
                        SELECT *
                        FROM {run}
                        WHERE {run_timestamp} >= (
                            SELECT COALESCE(MAX({summary}.{last_run_timestamp}), '1960-01-01'::timestamp)
                            FROM {summary}
                        )
                        AND NOT EXISTS (SELECT 1 FROM {summary} WHERE {summary}.{last_run_timestamp} >= {run}.{run_timestamp})
                        ORDER BY {run}.{run_timestamp} ASC
                    ) {run}
                    CROSS JOIN LATERAL (
                        SELECT
                            {selection_columns}
                        FROM
                            {experiment} {selection}
                        WHERE
                            {selection}.{experiment_id} = {run}.{experiment_id}
                        FOR UPDATE SKIP LOCKED
                    ) {selection}
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
            selection=selection_table,
            last_run_timestamp=summary.last_run_timestamp.identifier,
            selection_columns=selection_columns.of(selection_table),
            run_columns=run_columns.of(run.identifier),
            experiment_id=experiment.experiment_id.identifier,
            run_experiment_id=run.experiment_id.identifier,
            run_timestamp=run.run_timestamp.identifier,
            run=run.identifier,
            summary=summary.identifier,
            experiment=experiment.identifier,
            experiment_limit=Literal(experiment_limit),
        )

        def make_update_progress_query(num_summaries: int) -> Composed:
            return SQL("""
INSERT INTO {summary}
(
    {summary_columns}
)
VALUES {summary_column_placeholders}
ON CONFLICT ({experiment_id}) DO UPDATE SET
    {update_clause}
;""").format(
                summary=summary.identifier,
                summary_columns=summary_columns.columns_sql,
                summary_column_placeholders=sql_comma.join(
                    [SQL('({})').format(summary_columns.placeholders)] *
                    num_summaries),
                experiment_id=summary.experiment_id.identifier,
                update_clause=sql_comma.join(
                    (SQL('{c}=EXCLUDED.{c}').format(c=c)
                     for c in summary_columns.identifiers)),
            )

        result_columns = ColumnGroup(
            summary.last_run_timestamp,
            selection_columns,
            run_columns,
        )

        with ConnectionManager(schema.credentials) as connection:
            with connection.transaction():

                with connection.cursor(binary=True) as cursor:
                    # lock experiments and get runs to summarize

                    summary_rows = []
                    experiment_id = None
                    runs = []
                    last_run_timestamp = None
                    most_recent_run = None
                    experiment_attrs = {}
                    experiment_properties = {}

                    def make_summary():
                        nonlocal most_recent_run, experiment_id
                        if len(runs) > 0:
                            summary_rows.append(
                                (experiment_id, most_recent_run,
                                 self._compute_summary(runs)))
                            runs.clear()

                    cursor.execute(lock_and_get_query, binary=True)
                    for row in cursor.fetchall():

                        def value_of(column: Column) -> Any:
                            return row[result_columns[column]]

                        if last_run_timestamp is None:
                            last_run_timestamp = value_of(
                                summary.last_run_timestamp)
                        else:
                            last_run_timestamp = max(
                                last_run_timestamp,
                                value_of(summary.last_run_timestamp))

                        row_uid = value_of(run.experiment_id)
                        if row_uid != experiment_id:
                            make_summary()
                            experiment_id = row_uid
                            experiment_attrs = schema.attribute_map.attribute_map_from_ids(
                                value_of(experiment.experiment_attrs))
                            experiment_properties = schema.attribute_map.attribute_map_from_ids(
                                value_of(experiment.experiment_properties))
                            most_recent_run = value_of(run.run_timestamp)

                        most_recent_run = max(
                            most_recent_run,  # type: ignore
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

                    make_summary()

                    num_summaries = len(summary_rows)
                    if num_summaries > 0:
                        # write summaries to database
                        cursor.execute(
                            make_update_progress_query(num_summaries),
                            list(
                                chain(*((
                                    last_run_timestamp,
                                    experiment_id,
                                    last_updated,
                                    *self._summary_to_bytes(schema, summary),
                                ) for experiment_id, last_updated, summary in
                                        summary_rows))),
                        )

        return UpdateExperimentSummaryResult(num_summaries)

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
