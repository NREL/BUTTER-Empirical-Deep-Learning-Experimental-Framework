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
WITH
to_summarize AS (
	SELECT
		experiment2.experiment_id,
		experiment2.experiment
	FROM
		experiment2
	WHERE TRUE
		AND EXISTS ( SELECT 1 FROM run_status WHERE TRUE
		AND run_status.status = 2
		AND run_status.summarized IS NULL
		AND run_status.experiment_id IS NOT NULL
				   AND run_status.experiment_id = experiment2.experiment_id)

	FOR UPDATE SKIP LOCKED
	LIMIT 10
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
            to_summarize=Identifier("_to_summarize"),
            run_status=run_status.identifier,
            experiment_id=run_status.experiment_id.identifier,
            experiment=experiment_table.identifier,
            status=run_status.status.identifier,
            summarized=run_status.summarized.identifier,
            experiment_column=experiment_table.experiment.identifier,
            run_data=run_data.identifier,
            command=run_data.command.identifier,
            history=history.identifier,
            history_column=history.history.identifier,
            id=run_status.id.identifier,
            status_complete=Literal(JobStatus.Complete.value),
            experiment_limit=Literal(self.experiment_limit),
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
{to_summarize} AS (
	SELECT
		{experiment_id},
		{experiment}
	FROM
		{experiment}
	WHERE TRUE
        AND EXISTS ( SELECT 1 FROM {run_status} WHERE TRUE
            AND {run_status}.{status} = {status_complete}
            AND {run_status}.{summarized} IS NULL
            AND {run_status}.{experiment_id} IS NOT NULL
            )
	LIMIT {experiment_limit}
	FOR UPDATE SKIP LOCKED
),
updated_runs AS (
UPDATE {run_status} SET
	{summarized} = 1
FROM
	{to_summarize}
WHERE TRUE
	AND {run_status}.{status} = 2 AND {run_status}.{summarized} IS NULL AND {run_status}.{experiment_id} IS NOT NULL
	AND {run_status}.{experiment_id} = {to_summarize}.{experiment_id}
)
SELECT
	{to_summarize}.{experiment_id},
	{to_summarize}.{experiment},
	{history}.{id},
	{history}.{history_column}
FROM
	{to_summarize}
	INNER JOIN {history} ON ({history}.{experiment_id} = {to_summarize}.{experiment_id})
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
