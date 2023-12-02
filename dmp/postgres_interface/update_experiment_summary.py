from dataclasses import dataclass
from itertools import chain
from pprint import pprint
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type
from jobqueue.connection_manager import ConnectionManager

from jobqueue.job import Job
from jobqueue.job_status import JobStatus
from psycopg import ClientCursor
from psycopg.sql import SQL, Composed, Identifier, Literal
from dmp.parquet_util import convert_bytes_to_dataframe
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.postgres_interface.update_experiment_summary_result import (
    UpdateExperimentSummaryResult,
)
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.task import Task
from dmp.task.task_result import TaskResult
from dmp.worker import Worker
from dmp.postgres_interface.postgres_interface_common import sql_comma
from dmp.common import flatten, marshal_type_key
from dmp.context import Context

_summarizer_map: Dict[str, Callable] = {}

"""
WITH histories AS (
	SELECT
		selected_experiment.experiment_id,
		run_data.command->'experiment' experiment,
		run_status.id,
		run_status.update_time,
		history.history
	FROM
		(
			SELECT DISTINCT ON (experiment_id) *
			FROM
			(
                SELECT
                    run_status.experiment_id,
                FROM
                    run_status
                    CROSS JOIN LATERAL (
                        SELECT rs.id root_id FROM
                            run_status rs
                        WHERE TRUE
                            AND rs.experiment_id = run_status.experiment_id
                            AND rs.status = 2
                        ORDER BY rs.update_time ASC, id
                        LIMIT 1) root_id
                    CROSS JOIN LATERAL (
                        SELECT rs.id FROM
                            run_status rs
                        WHERE rs.id = root_id
                        FOR UPDATE SKIP LOCKED
                    ) root_run
                WHERE TRUE
                    AND run_status.status = 2
                    AND NOT EXISTS (
                        SELECT 1 FROM experiment2
                        WHERE
                            experiment2.experiment_id = run_status.experiment_id
                            AND experiment2.most_recent_run >= run_status.update_time
                    )
                ORDER BY run_status.update_time DESC
                LIMIT 10
			) selected_experiment
		) selected_experiment
		INNER JOIN run_status ON (run_status.experiment_id = selected_experiment.experiment_id AND run_status.status = 2)
		INNER JOIN run_data ON (run_data.id = run_status.id)
		INNER JOIN history ON (history.id = run_status.id)
	ORDER BY experiment_id
), claim AS (
	INSERT INTO experiment2 (experiment_id, experiment, most_recent_run, num_runs)
	SELECT
		experiment_id, experiment, MAX(update_time) most_recent_run, COUNT(1) num_runs
	FROM
		histories
	GROUP BY experiment_id, experiment
	ON CONFLICT (experiment_id) DO UPDATE SET
		most_recent_run = EXCLUDED.most_recent_run,
		num_runs = EXCLUDED.num_runs
)
SELECT * FROM histories
;

"""


@dataclass
class UpdateExperimentSummary(Task):
    experiment_limit: int
    lock_limit: int

    @staticmethod
    def register_types(types: Iterable[Type]) -> None:
        for target_type in types:
            UpdateExperimentSummary.register_type(target_type)

    @staticmethod
    def register_type(target_type: Type) -> None:
        type_code = target_type.__name__
        _summarizer_map[type_code] = target_type.summarize

    def __call__(
        self,
        context: Context,
    ) -> TaskResult:
        from dmp.marshaling import marshal

        schema = context.schema
        experiment_table = schema.experiment
        run_status = schema.run_status
        run_data = schema.run_data
        history = schema.history

        format_args = dict(
            history=history.identifier,
            run_status=run_status.identifier,
            run_data=run_data.identifier,
            experiment=experiment_table.identifier,
            id=run_status.id.identifier,
            experiment_id=run_status.experiment_id.identifier,
            update_time=run_status.update_time.identifier,
            history_column=history.history.identifier,
            command=run_data.command.identifier,
            experiment_column=experiment_table.experiment.identifier,
            most_recent_run=experiment_table.most_recent_run.identifier,
            status=run_status.status.identifier,
            num_runs=experiment_table.num_runs.identifier,
            selected_experiment=Identifier("_selected_experiment"),
            root_run=Identifier("_root_run"),
            root_selection=Identifier("_root_selection"),
            root_id=Identifier("_root_id"),
            root_lock=Identifier("_root_lock"),
            status_complete=Literal(JobStatus.Complete.value),
            experiment_limit=Literal(self.experiment_limit),
            lock_limit=Literal(self.lock_limit),
            runs_to_update=Identifier("_runs_to_update"),
            by_epoch=experiment_table.by_epoch.identifier,
            by_loss=experiment_table.by_loss.identifier,
        )

        claim_columns = ColumnGroup(
            run_status.experiment_id,
            experiment_table.experiment,
            run_status.id,
            run_status.update_time,
            history.history,
        )

        claim_and_get_query = SQL(
            """
WITH {runs_to_update} AS (
	SELECT
		{selected_experiment}.{experiment_id},
		{run_data}.{command}->'experiment' {experiment_column},
		{run_status}.{id},
		{run_status}.{update_time},
		{history}.{history_column}
	FROM
        (
            SELECT DISTINCT ON ({experiment_id}) *
            FROM
            (
                SELECT
                    {run_status}.{experiment_id}
                FROM
                    {run_status}
                    CROSS JOIN LATERAL (
                        SELECT {root_selection}.{id} {root_id} FROM
                            {run_status} {root_selection}
                        WHERE TRUE
                            AND {root_selection}.{experiment_id} = {run_status}.{experiment_id}
                            AND {root_selection}.{status} = {status_complete}
                        ORDER BY {root_selection}.{update_time} ASC, {root_selection}.{id}
                        LIMIT 1
                    ) {root_id}
                    CROSS JOIN LATERAL (
                        SELECT {root_selection}.{id} {root_lock} FROM
                            {run_status} {root_selection}
                        WHERE {root_selection}.{id} = {root_id}.{root_id}
                    ) {root_lock}
                WHERE TRUE
                    AND {run_status}.{status} = {status_complete}
                    AND NOT EXISTS (
                        SELECT 1 FROM {experiment}
                        WHERE
                            {experiment}.{experiment_id} = {run_status}.{experiment_id}
                            AND {experiment}.{most_recent_run} >= {run_status}.{update_time}
                    )
                ORDER BY {run_status}.{update_time} DESC
                LIMIT {lock_limit}
            ) {selected_experiment}
            LIMIT {experiment_limit}
        ) {selected_experiment}
		INNER JOIN {run_status} ON ({run_status}.{experiment_id} = {selected_experiment}.{experiment_id} AND {run_status}.{status} = {status_complete})
        INNER JOIN {run_data} ON ({run_data}.{id} = {run_status}.{id})
		INNER JOIN {history} ON ({history}.{id} = {run_status}.{id})
    ORDER BY {experiment_id}
), claim AS (
	INSERT INTO {experiment} ({experiment_id}, {experiment_column}, {most_recent_run}, {num_runs})
	SELECT
		{experiment_id}, {experiment_column}, MAX({update_time}) {most_recent_run}, COUNT(1) {num_runs}
	FROM
		{runs_to_update}
	GROUP BY {experiment_id}, {experiment_column}
	ON CONFLICT ({experiment_id}) DO UPDATE SET
		{most_recent_run} = EXCLUDED.{most_recent_run},
		{num_runs} = EXCLUDED.{num_runs},
        {by_epoch} = NULL,
        {by_loss} = NULL
)
SELECT * FROM {runs_to_update}
;"""
        ).format(**format_args)

        rows = []
        with ConnectionManager(schema.credentials) as connection:
            # with ClientCursor(connection) as cursor:
            #     print(cursor.mogrify(claim_and_get_query))
            with connection.cursor(binary=True) as cursor:
                cursor.execute(claim_and_get_query, binary=True)
                rows = cursor.fetchall()

        result = UpdateExperimentSummaryResult(0, 0)
        encoded_histories = []
        experiment_id = None
        marshaled_experiment = None

        def summarize_experiment():
            nonlocal result, encoded_histories, experiment_id, marshaled_experiment
            if len(encoded_histories) == 0 or marshaled_experiment is None:
                return
            try:
                # pprint(marshaled_experiment)
                experiment = marshal.demarshal(marshaled_experiment)
                # print(f"demarshaled experiment: {experiment}")
                histories = [
                    (run_id, convert_bytes_to_dataframe(history))
                    for run_id, history in encoded_histories
                ]

                schema.store_summary(
                    experiment,
                    experiment_id,  # type: ignore
                    experiment.summarize(histories),
                )
                result.num_experiments_updated += 1
            except Exception as e:
                result.num_experiments_excepted += 1
                print(f"Error summarizing experiment {experiment_id}: {e}")
                import traceback

                traceback.print_exc()

        for row in rows:

            def value_of(column: Column) -> Any:
                return row[claim_columns[column]]

            this_experiment_id = value_of(run_status.experiment_id)
            if this_experiment_id != experiment_id:
                summarize_experiment()
                encoded_histories.clear()
                experiment_id = this_experiment_id
                marshaled_experiment = value_of(experiment_table.experiment)

            encoded_histories.append(
                (value_of(run_status.id), value_of(history.history))
            )
        summarize_experiment()

        return result
