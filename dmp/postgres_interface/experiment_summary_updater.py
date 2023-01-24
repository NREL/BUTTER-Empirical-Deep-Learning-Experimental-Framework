from collections import Callable
from dataclasses import dataclass
from typing import Dict, Iterable, List, Type
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
from dmp.postgres_interface.postgres_interface_common import comma_sql
from dmp.common import flatten, marshal_type_key

_summarizer_map: Dict[str, Callable[[Iterable[ExperimentResultRecord]],
                                    ExperimentSummaryRecord]] = {}


@dataclass
class ExperimentSummaryUpdater(Task):

    def __call__(self, worker: Worker, job: Job) -> TaskResult:
        schema = worker._schema

        experiment_limit = 16

        experiment_table = schema.experiment
        experiment_attrs = experiment_table['attrs']

        run_table = schema.run
        experiment_summary_progress_table = schema.experiment_summary_progress
        experiment_summary_table = schema.experiment_summary

        run_experiment_uid = run_table['experiment_uid']
        experiment_uid = experiment_table['uid']

        run_timestamp = run_table['timestamp']
        last_updated = experiment_summary_progress_table['last_updated']

        summary_data_columns = experiment_summary_table['data']
        summary_id_column = experiment_summary_table['experiment_uid']

        summary_columns = ColumnGroup.concatenate((
            summary_id_column,
            summary_data_columns,
        ))

        all_summary_columns = experiment_summary_table[
            'experiment_uid'] + summary_columns

        selection_table = Identifier('_selection')

        run_columns = ColumnGroup.concatenate((
            run_table['experiment_uid'],
            run_table['values'],
            run_table['data'],
            run_table['history'],
        ))

        # lock_table = Identifier('_experiment')
        lock_and_get_query = SQL("""
SELECT
    {experiment_table}.*
FROM
    (
        SELECT * FROM
            (
                SELECT {run_experiment_uid}
                FROM
                    (
                        SELECT 
                            {run_timestamp}, 
                            {run_experiment_uid}
                        FROM 
                            {run_table}
                        WHERE 
                            {run_timestamp} >= (
                                SELECT {last_updated} 
                                FROM {experiment_summary_progress_table} 
                                LIMIT 1
                                )
                        ORDER BY {run_timestamp} ASC, {run_experiment_uid}
                        LIMIT {experiment_scan_limit}
                    ) {selection_table}
                GROUP BY {run_experiment_uid}
                LIMIT {experiment_limit}
            ) {selection_table}
        ORDER BY {run_experiment_uid}
    ) {selection_table}
    CROSS JOIN LATERAL (
        SELECT
            {experiment_table}.{experiment_uid},
            {experiment_table}.{experiment_attrs}
        FROM
            {experiment_table}
        WHERE
            {experiment_table}.{experiment_uid} = {selection_table}.{run_experiment_uid}
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    ) e
;""").format(
            experiment_table=experiment_table.name_sql,
            run_experiment_uid=run_experiment_uid.column_identifiers,
            run_timestamp=run_timestamp.column_identifiers,
            run_table=run_table.name_sql,
            last_updated=last_updated.column_identifiers,
            experiment_summary_progress_table=experiment_summary_progress_table
            .name_sql,
            experiment_scan_limit=Literal(experiment_limit * 8),
            selection_table=selection_table,
            experiment_limit=Literal(experiment_limit),
            experiment_uid=experiment_uid.column_identifiers,
            experiment_attrs=experiment_attrs.column_identifiers,
        )

        get_runs_query = SQL("""
SELECT
    {run_columns}
FROM
    {run_table}
WHERE
    {run_experiment_uid} = %s
;""").format(
            run_columns=run_columns.column_identifiers,
            run_table=run_table.name_sql,
            run_experiment_uid=run_experiment_uid.column_identifiers,
        )

        def make_update_progress_query(num_summaries: int) -> Composed:
            return SQL("""
WITH {inserted} AS (
    SELECT MAX({last_updated}) {last_updated}
    FROM
    (
        INSERT INTO {experiment_summary_table}
        (
            {summary_columns}
        )
        VALUES {summary_column_placeholders}
        ON CONFLICT ({experiment_uid}) DO UPDATE SET
            {update_clause}
        )
        RETURNING {last_updated}
    ) {inserted}
)
UPDATE {experiment_summary_progress_table} SET 
    {last_updated_progress} = (SELECT {last_updated} FROM {inserted} LIMIT 1)
WHERE 
    {last_updated_progress} < (SELECT {last_updated} FROM {inserted} LIMIT 1)
;""").format(
                inserted=Identifier('_inserted'),
                experiment_summary_table=experiment_summary_table.name_sql,
                last_updated=experiment_summary_table['last_updated'].
                column_identifiers,
                summary_columns=all_summary_columns.column_identifiers,
                summary_column_placeholders=comma_sql.join(
                    [SQL('({})').format(all_summary_columns.placeholders)] *
                    num_summaries),
                experiment_uid=experiment_summary_table['experiment_uid'],
                values_table=Identifier('_values'),
                update_clause=comma_sql.join(
                    (SQL('{c}=EXCLUDED.{c}').format(c=c)
                     for c in summary_columns.column_identifiers)),
                experiment_summary_progress_table=
                experiment_summary_progress_table,
                last_updated_progress=experiment_summary_progress_table[
                    'last_updated'].column_identifiers,
            )


#         write_summary_query = SQL("""
# INSERT INTO {experiment_summary_table}
#     (
#         {last_updated},
#         {summary_columns}
#     )
#     (
#         SELECT
#             CURRENT_TIMESTAMP() {last_updated},
#             *
#         FROM
#             (VALUES ({summary_column_placeholders})) v ({summary_columns})
#     ) v
# ON CONFLICT ({experiment_uid}) DO UPDATE SET
#     {last_updated} = CURRENT_TIMESTAMP(),
#     {update_clause}
# )
# ;""").format(experiment_summary_table=experiment_summary_table.name_sql,
#              last_updated=experiment_summary_table['last_updated'].
#              column_identifiers,
#              summary_columns=summary_columns.columns_sql,
#              summary_column_placeholders=summary_columns.placeholders,
#              experiment_uid=experiment_summary_table['experiment_uid'],
#              update_clause=comma_sql.join(
#                  (SQL('{c}=EXCLUDED.{c}').format(c=c)
#                   for c in summary_columns.column_identifiers)))

#         update_progress_query = SQL("""
# UPDATE {experiment_summary_progress_table}
#     SET {last_updated} = CURRENT_TIMESTAMP()
# WHERE {last_updated} < CURRENT_TIMESTAMP()
# ;""").format(
#             experiment_summary_progress_table=experiment_summary_progress_table
#             .name_sql,
#             last_updated=last_updated,
#         )

        with ConnectionManager(schema.credentials) as connection:
            with connection.transaction():
                with connection.cursor(binary=True) as cursor:

                    # lock and get experiments
                    experiments = []
                    cursor.execute(lock_and_get_query, binary=True)
                    for row in cursor.fetchall():
                        experiments.append((
                            row[0],
                            row[1],
                        ))

                    num_experiments = len(experiments)
                    if num_experiments <= 0:
                        return TaskResult()

                    # compute summaries
                    summaries: List[ExperimentSummaryRecord] = []
                    for experiment_uid, experiment_attr_ids in experiments:

                        # reconstitute experiment attributes
                        experiment_attrs = schema.attribute_map.attribute_map_from_ids(
                            experiment_attr_ids)

                        # get all runs for this experiment
                        runs: List[ExperimentResultRecord] = []
                        cursor.execute(get_runs_query, binary=True)
                        for row in cursor.fetchall():

                            run_data = row[run_columns['json']]
                            for c in run_table['value'].columns:
                                run_data[c] = row[run_columns[c]]

                            run_history = row['run_history']

                            runs.append(
                                ExperimentResultRecord(
                                    experiment_attrs,
                                    run_data,
                                    run_history,
                                ))

                        # call summarizer for this experiment
                        experiment_type_code = experiment_attrs[
                            marshal_type_key]
                        summaries.append(
                            _summarizer_map[experiment_type_code](runs))

                    # write summaries to database
                    cursor.execute(
                        make_update_progress_query(num_experiments),
                        list(
                            flatten(
                                ((
                                    s.experiment_uid,
                                    s.core_data,
                                    s.extended_data,
                                ) for s in summaries),
                                levels=1,
                            )),
                    )

        return TaskResult()

    @staticmethod
    def register_types(types: Iterable[Type]) -> None:
        for target_type in types:
            ExperimentSummaryUpdater.register_type(target_type)

    @staticmethod
    def register_type(target_type: Type) -> None:
        type_code = target_type.__name__
        _summarizer_map[type_code] = target_type.summarize