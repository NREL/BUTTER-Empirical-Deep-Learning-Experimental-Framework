from dataclasses import dataclass
from typing import Iterable, Type
from jobqueue.connect import load_credentials
from jobqueue.connection_manager import ConnectionManager

from jobqueue.job import Job
from psycopg import sql
from dmp.task.task import Task
from dmp.task.task_result import TaskResult
from dmp.worker import Worker

_summarizer_map = {}


@dataclass
class ExperimentSummaryUpdater(Task):

    def __call__(self, worker: Worker, job: Job) -> TaskResult:
        comma = sql.SQL(',')
        experiment_limit = 16
        experiment_table = sql.Identifier('experiment2')
        run_table = sql.Identifier('run2')
        experiment_summary_table = sql.Identifier('experiment_summary')
        attr_table = sql.Identifier('attr')
        experiment_summary_progress_table = sql.Identifier(
            'experiment_summary_progress')

        experiment_summary_columns = [
            'experiment_uid', 'core_data', 'extended_data'
        ]

        lock_and_get_query = sql.SQL("""
SELECT
    e.*
FROM
    (
        SELECT * FROM
            (
                SELECT experiment_uid
                FROM
                    (
                        SELECT 
                            run_timestamp, experiment_uid
                        FROM 
                            {run_table} r
                        WHERE 
                            r.run_timestamp >= (SELECT last_updated FROM {experiment_summary_progress_table} LIMIT 1)
                        ORDER BY run_timestamp, experiment_uid
                        LIMIT {experiment_scan_limit}
                    ) rx
                GROUP BY experiment_uid
                LIMIT {experiment_limit}
            ) rx
        ORDER BY experiment_uid
    ) rx
    CROSS JOIN LATERAL (
        SELECT
            e.experiment_uid,
            e.experiment_attrs
        FROM
            {experiment_table} e
        WHERE
            e.experiment_uid = rx.experiment_uid
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    ) e
;""").format(
            run_table=run_table,
            experiment_summary_progress_table=experiment_summary_progress_table,
            experiment_scan_limit=sql.Literal(experiment_limit * 8),
            experiment_limit=sql.Literal(experiment_limit),
            experiment_table=experiment_table,
        )

        get_runs_query = sql.SQL("""
SELECT
    *
FROM
    {run_table} r
WHERE
    r.experiment_uid = %s
;""").format(run_table=run_table, )

        write_summary_query = sql.SQL("""
INSERT INTO {experiment_summary_table} s (last_updated, {summary_columns})
    (
        SELECT 
            CURRENT_TIMESTAMP() last_updated, 
            *
        FROM 
            (VALUES (%s,%s,%s)) v ({summary_columns})
    ) v
ON CONFLICT (experiment_uid) DO UPDATE SET
    last_updated = CURRENT_TIMESTAMP(),
    {update_clause}
)
;""").format(experiment_summary_table=experiment_summary_table,
             summary_columns=comma.join(
                 (sql.Identifier(c) for c in experiment_summary_columns)),
             update_clause=comma.join(
                 (sql.SQL('{c}=EXCLUDED.{c}').format(c=sql.Identifier(c))
                  for c in experiment_summary_columns)))

        update_progress_query = sql.SQL("""
UPDATE experiment_summary_progress
    SET last_updated = CURRENT_TIMESTAMP()
WHERE last_updated < CURRENT_TIMESTAMP()
;""")
        credentials = load_credentials('dmp')
        with ConnectionManager(credentials) as connection:
            with connection.transaction():
                results = []

                experiment_run_data = []
                current_experiment_uid = None
                current_task_type = None

                def do_accumulate():
                    nonlocal experiment_run_data, current_experiment_uid, current_task_type
                    if current_task_type is None:
                        return

                    summary_result = _summarizer_map[current_task_type](
                        experiment_run_data)
                    results.append((
                        experiment_uid,
                        summary_result.core_data,
                        summary_result.extended_data,
                    ))

                with connection.cursor(binary=True) as cursor:
                    cursor.execute(lock_and_get_query, binary=True)

                    for row in cursor:
                        experiment_uid = row[0]
                        task_type = row[1]

                        if current_experiment_uid != experiment_uid:
                            do_accumulate()
                            experiment_run_data = []
                            current_experiment_uid = experiment_uid
                            current_task_type = task_type

                        experiment_run_data.append()

                    do_accumulate()

                    cursor.executemany(write_summary_query,
                                       results,
                                       returning=False)
                    cursor.execute(update_progress_query)

        return TaskResult()

    @staticmethod
    def register_types(types: Iterable[Type]) -> None:
        for target_type in types:
            ExperimentSummaryUpdater.register_type(target_type)

    @staticmethod
    def register_type(target_type: Type) -> None:
        type_code = target_type.__name__
        _summarizer_map[type_code] = target_type.make_summary