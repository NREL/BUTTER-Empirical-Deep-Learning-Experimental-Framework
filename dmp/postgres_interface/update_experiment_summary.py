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
WITH experiments_to_update AS (
	SELECT
		experiment_id,
		experiment
	FROM
		(
			SELECT experiment2.experiment_id
			FROM
				experiment2
			WHERE TRUE
				AND EXISTS (
					SELECT 1 FROM run_status
					WHERE TRUE
						AND experiment2.experiment_id = run_status.experiment_id
						AND run_status.status = 2
						AND run_status.summarized IS NULL
						AND run_status.experiment_id IS NOT NULL)
		) target
		CROSS JOIN LATERAL (
			SELECT experiment_lock.experiment FROM experiment2 experiment_lock
			WHERE experiment_lock.experiment_id = target.experiment_id
			FOR UPDATE OF experiment_lock SKIP LOCKED
		) x
	LIMIT 10
),
new_experiments AS (
	INSERT INTO experiment2 (experiment_id, experiment)
	SELECT
		selected_experiment.experiment_id,
		run_data.command->'experiment' experiment
	FROM
	(
		SELECT DISTINCT ON (run_status.experiment_id) run_status.experiment_id, run_status.id
		FROM
			run_status
		WHERE TRUE
			AND status = 2
			AND summarized IS NULL
			AND run_status.experiment_id IS NOT NULL
			AND NOT EXISTS (SELECT 1 FROM experiment2 WHERE experiment2.experiment_id = run_status.experiment_id)
		LIMIT GREATEST(0, 10 - (SELECT COUNT(1) FROM experiments_to_update))
	) selected_experiment
	INNER JOIN run_data ON (run_data.id = selected_experiment.id)
	ON CONFLICT (experiment_id) DO NOTHING
	RETURNING experiment_id, experiment
),
to_summarize AS (
	SELECT * FROM
	(
		SELECT * FROM experiments_to_update
		UNION ALL
		SELECT * FROM new_experiments
	) experiment_ids
	ORDER BY experiment_id
),
updated_runs AS (
UPDATE run_status SET
	summarized = 1
FROM
	to_summarize
WHERE TRUE
	AND run_status.status = 2 AND run_status.summarized IS NULL AND run_status.experiment_id IS NOT NULL
	AND run_status.experiment_id = to_summarize.experiment_id
)
SELECT
	to_summarize.experiment_id,
	to_summarize.experiment,
	history.id,
	history.history
FROM
	to_summarize
	INNER JOIN history ON (history.experiment_id = to_summarize.experiment_id)
ORDER BY experiment_id
;

"""


@dataclass
class UpdateExperimentSummary(Task):
    experiment_limit: int
    insert_limit: int

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
            experiments_to_update=Identifier("_experiments_to_update"),
            experiment_id=run_status.experiment_id.identifier,
            experiment=experiment_table.identifier,
            run_status=run_status.identifier,
            status=run_status.status.identifier,
            status_complete=Literal(JobStatus.Complete.value),
            summarized=run_status.summarized.identifier,
            experiment_lock=Identifier("_experiment_lock"),
            target=Identifier("_target"),
            new_experiments=Identifier("_new_experiments"),
            experiment_column=experiment_table.experiment.identifier,
            selected_experiment=Identifier("_selected_experiment"),
            run_data=run_data.identifier,
            command=run_data.command.identifier,
            to_summarize=Identifier("_to_summarize"),
            experiment_ids=Identifier("_experiment_ids"),
            updated_runs=Identifier("_updated_runs"),
            history=history.identifier,
            history_column=history.history.identifier,
            id=run_status.id.identifier,
            experiment_limit=Literal(self.experiment_limit),
            insert_limit=Literal(self.insert_limit),
        )

        claim_columns = ColumnGroup(
            experiment_table.experiment_id,
            experiment_table.experiment,
            run_status.id,
            history.history,
        )

        claim_and_get_query = SQL(
            """
WITH
{new_experiments} AS (
	INSERT INTO {experiment} ({experiment_id}, {experiment_column})
	SELECT
		{selected_experiment}.{experiment_id},
		{run_data}.{command}->'experiment' {experiment_column}
	FROM
	(
        SELECT DISTINCT ON ({experiment_id}) {experiment_id}, {id}
        FROM
            {run_status}
        WHERE TRUE
            AND {status} = {status_complete}
            AND {summarized} IS NULL
            AND {run_status}.{experiment_id} IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM {experiment} WHERE {experiment}.{experiment_id} = {run_status}.{experiment_id})
        LIMIT {insert_limit}
	) {selected_experiment}
	INNER JOIN {run_data} ON ({run_data}.{id} = {selected_experiment}.{id})
	ON CONFLICT ({experiment_id}) DO NOTHING
	RETURNING {experiment_id}, {experiment_column}
),
{experiments_to_update} AS (
	SELECT
		{experiment_id}, {experiment_column}
	FROM
		(
			SELECT {experiment}.{experiment_id}
			FROM
				{experiment}
			WHERE TRUE
				AND EXISTS (
					SELECT 1 FROM {run_status}
					WHERE TRUE
						AND {experiment}.{experiment_id} = {run_status}.{experiment_id}
						AND {run_status}.{status} = {status_complete}
						AND {run_status}.{summarized} IS NULL
						AND {run_status}.{experiment_id} IS NOT NULL)
		) {target}
		CROSS JOIN LATERAL (
			SELECT {experiment_lock}.{experiment_column} FROM {experiment} {experiment_lock}
			WHERE {experiment_lock}.{experiment_id} = {target}.{experiment_id}
			FOR UPDATE OF {experiment_lock} SKIP LOCKED
		) x
	LIMIT GREATEST(0, {experiment_limit} - (SELECT COUNT(1) FROM {new_experiments}))
),
{to_summarize} AS (
	SELECT * FROM
	(
		SELECT * FROM {experiments_to_update}
		UNION ALL
		SELECT * FROM {new_experiments}
	) {experiment_ids}
	ORDER BY {experiment_id}
    LIMIT {experiment_limit}
),
{updated_runs} AS (
UPDATE {run_status} SET
	{summarized} = 1
FROM
	{to_summarize}
WHERE TRUE
	AND {run_status}.{status} = {status_complete}
    AND {run_status}.{summarized} IS NULL
    AND {run_status}.{experiment_id} IS NOT NULL
	AND {run_status}.{experiment_id} = {to_summarize}.{experiment_id}
)
SELECT
	{to_summarize}.{experiment_id},
    {to_summarize}.{experiment_column},
	{history}.{id},
	{history}.{history_column}
FROM
	{to_summarize}
	INNER JOIN {history} ON ({history}.{experiment_id} = {to_summarize}.{experiment_id})
    INNER JOIN {run_data} ON ({run_data}.{id} = {history}.{id})
ORDER BY {experiment_id}
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

        summaries = []

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
                summaries.append(
                    (
                        experiment_id,  # type: ignore
                        experiment.summarize(histories),
                    )
                )
                del histories
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

        try:
            schema.store_summaries(summaries)
        except Exception as e:
            result.num_experiments_excepted += len(summaries)
            result.num_experiments_updated -= len(summaries)
            print(f"Error storing summaries: {e}")
            import traceback

            traceback.print_exc()

        return result
