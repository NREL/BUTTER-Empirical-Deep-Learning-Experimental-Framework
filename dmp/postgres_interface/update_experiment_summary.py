from collections import Callable
from dataclasses import dataclass
import io
from typing import Dict, Iterable, List, Sequence, Type
from jobqueue.connect import load_credentials
from jobqueue.connection_manager import ConnectionManager

from jobqueue.job import Job
from psycopg.sql import SQL, Composed, Identifier, Literal
from dmp.postgres_interface.column_group import ColumnGroup
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

    def __call__(self, worker: Worker, job: Job) -> TaskResult:
        schema = worker._schema

        experiment_limit = 64

        experiment_table = schema.experiment
        experiment_attrs_group = experiment_table['attrs']
        experiment_properties_group = experiment_table['properties']

        run_table = schema.run
        experiment_summary_table = schema.experiment_summary

        experiment_uid = schema.experiment_id_group

        # run_timestamp = run_table['timestamp']
        last_run_timestamp_group = experiment_summary_table[
            'last_run_timestamp']

        run_columns = ColumnGroup.concatenate((
            experiment_uid,
            run_table['timestamp'],
            run_table['values'],
            run_table['data'],
            run_table['history'],
        ))

        result_columns = ColumnGroup.concatenate((
            last_run_timestamp_group,
            experiment_attrs_group,
            experiment_properties_group,
            run_columns,
        ))

        experiment_summary_columns = ColumnGroup.concatenate((
            last_run_timestamp_group,
            experiment_uid,
            experiment_summary_table['data'],
        ))

        selection_table = Identifier('_selection')

        lock_and_get_query = SQL("""
SELECT
    {selection_table}.{last_run_timestamp},
    {selection_table}.{experiment_attrs},
    {selection_table}.{experiment_properties},
    {run_columns}
FROM
    (
        SELECT DISTINCT ON({experiment_uid})
            {experiment_uid},
            {experiment_attrs},
            {experiment_properties},
            MAX({run_timestamp}) OVER (PARTITION BY {experiment_uid}) {last_run_timestamp}
        FROM
            (
                SELECT 
                    {run_table}.{run_timestamp}, 
                    {experiment_table}.{experiment_uid},
                    {experiment_table}.{experiment_attrs},
                    {experiment_table}.{experiment_properties}
                FROM 
                    {run_table} LEFT JOIN {experiment_summary_table} ON (
                        {run_table}.{experiment_uid} = {experiment_summary_table}.{experiment_uid}
                        AND {experiment_summary_table}.{last_run_timestamp} > {run_timestamp})
                    CROSS JOIN LATERAL (
                        SELECT
                            {experiment_table}.{experiment_uid},
                            {experiment_table}.{experiment_attrs},
                            {experiment_table}.{experiment_properties}
                        FROM
                            {experiment_table}
                        WHERE
                            {experiment_table}.{experiment_uid} = {run_table}.{experiment_uid}
                            AND experiment_properties @> array[2568] and experiment_attrs @> array[75, 224]
                        FOR UPDATE SKIP LOCKED
                    ) {experiment_table}
                WHERE 
                    {run_timestamp} > (
                        SELECT COALESCE(MAX({experiment_summary_table}.{run_update_limit}), '1960-01-01'::timestamp)
                        FROM {experiment_summary_table}
                        )
                    AND {experiment_summary_table}.{experiment_uid} IS NULL
                ORDER BY {run_table}.{run_timestamp} ASC
                LIMIT {experiment_limit}
            ) {selection_table}
    ) {selection_table}
    CROSS JOIN LATERAL (
        SELECT
            {run_columns}
        FROM 
            {run_table} 
        WHERE 
            {run_table}.{experiment_uid} = {selection_table}.{experiment_uid}
    ) {run_table}
;""").format(
            selection_table=selection_table,
            last_run_timestamp=last_run_timestamp_group.identifier,
            experiment_attrs=experiment_attrs_group.identifier,
            experiment_properties=experiment_properties_group.identifier,
            run_columns=run_columns.of(run_table.identifier),
            experiment_uid=experiment_uid.identifier,
            run_timestamp=run_table['timestamp'].identifier,
            run_table=run_table.identifier,
            experiment_summary_table=experiment_summary_table.identifier,
            experiment_table=experiment_table.identifier,
            run_update_limit=experiment_summary_table['run_update_limit'].
            identifier,
            experiment_limit=Literal(experiment_limit),
        )

        def make_update_progress_query(num_summaries: int) -> Composed:
            return SQL("""
INSERT INTO {experiment_summary_table}
(
    {summary_columns}
)
VALUES {summary_column_placeholders}
ON CONFLICT ({experiment_uid}) DO UPDATE SET
    {update_clause}
;""").format(
                experiment_summary_table=experiment_summary_table.identifier,
                summary_columns=experiment_summary_columns.identifiers,
                summary_column_placeholders=sql_comma.join([
                    SQL('({})').format(experiment_summary_columns.placeholders)
                ] * num_summaries),
                experiment_uid=experiment_uid,
                update_clause=sql_comma.join(
                    (SQL('{c}=EXCLUDED.{c}').format(c=c)
                     for c in experiment_summary_columns.identifiers)),
            )

        with ConnectionManager(schema.credentials) as connection:
            with connection.transaction():
                with connection.cursor(binary=True) as cursor:

                    # lock experiments and get runs to summarize
                    summaries = []
                    experiment_uid = None
                    runs = []
                    last_updated = None
                    experiment_attrs = {}
                    experiment_properties = {}

                    cursor.execute(lock_and_get_query, binary=True)
                    for row in cursor.fetchall():
                        row_uid = row[result_columns[
                            schema.experiment_id_column]]
                        if row_uid != experiment_uid:
                            if len(runs) > 0:
                                summaries.append((last_updated,
                                                  self._compute_summary(runs)))

                            experiment_uid = row_uid
                            runs.clear()
                            experiment_attrs = schema.attribute_map.attribute_map_from_ids(
                                row[result_columns[
                                    experiment_attrs_group.column]])
                            experiment_properties = schema.attribute_map.attribute_map_from_ids(
                                row[result_columns[
                                    experiment_properties_group.column]])
                            last_updated = row[result_columns['run_timestamp']]

                        last_updated = max(
                            last_updated,  # type: ignore
                            row[result_columns['run_timestamp']],
                        )
                        run_data = row[result_columns['run_data']]
                        for c in run_table['values'].columns:
                            run_data[c] = row[result_columns[c]]

                        run_history = schema.load_history_from_bytes(
                            row[result_columns['run_history']]
                            )

                        runs.append(
                            ExperimentResultRecord(
                                experiment_attrs,
                                experiment_properties,
                                run_data,
                                run_history, # type: ignore
                                None,
                            ))

                    if len(runs) > 0:
                        summaries.append(
                            (last_updated, self._compute_summary(runs)))

                    # num_summaries = len(summaries)
                    # if num_summaries > 0:
                    #     # write summaries to database
                    #     cursor.execute(
                    #         make_update_progress_query(num_summaries),
                    #         list(
                    #             flatten(
                    #                 ((
                    #                     last_updated,
                    #                     summary.experiment_uid,
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

    @staticmethod
    def register_types(types: Iterable[Type]) -> None:
        for target_type in types:
            UpdateExperimentSummary.register_type(target_type)

    @staticmethod
    def register_type(target_type: Type) -> None:
        type_code = target_type.__name__
        _summarizer_map[type_code] = target_type.summarize