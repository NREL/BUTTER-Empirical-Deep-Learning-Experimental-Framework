from dataclasses import dataclass
from itertools import chain
from pprint import pprint
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type
from jobqueue.connection_manager import ConnectionManager


from psycopg import ClientCursor
from psycopg.sql import SQL, Composed, Identifier, Literal
from dmp.parquet_util import convert_bytes_to_dataframe
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.postgres_interface.update_experiment_summary_result import (
    UpdateExperimentSummaryResult,
)
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.run_table import RunStatus
from dmp.task.run_command import RunCommand
from dmp.task.run_result import RunResult
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
		AND EXISTS ( SELECT 1 FROM run_table WHERE TRUE
		AND run_table.status = 2
		AND run_table.summarized IS NULL
		AND run_table.experiment_id IS NOT NULL
				   AND run_table.experiment_id = experiment2.experiment_id)

	FOR UPDATE SKIP LOCKED
	LIMIT 10
),
updated_runs AS (
UPDATE run_table SET
	summarized = 1
FROM
	to_summarize
WHERE TRUE
	AND run_table.status = 2 AND run_table.summarized IS NULL AND run_table.experiment_id IS NOT NULL
	AND run_table.experiment_id = to_summarize.experiment_id
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
class UpdateExperimentSummary(RunCommand):
    experiment_limit: int

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
    ) -> RunResult:
        from dmp.marshaling import marshal

        database = context.database
        experiment_table = database._experiment_table
        run_table = database._run_table

        format_args = dict(
            to_summarize=Identifier("_to_summarize"),
            run_table=run_table.identifier,
            experiment_id=run_table.experiment_id.identifier,
            experiment=experiment_table.identifier,
            status=run_table.status.identifier,
            experiment_column=experiment_table.experiment.identifier,
            command=run_table.command.identifier,
            history_column=run_table.history.identifier,
            id=run_table.id.identifier,
            status_complete=Literal(RunStatus.Complete.value),
            status_summarized=Literal(RunStatus.Summarized.value),
            experiment_limit=Literal(self.experiment_limit),
        )

        result_columns = ColumnGroup(
            experiment_table.experiment_id,
            experiment_table.experiment,
            run_table.id,
            run_table.history,
        )

        claim_and_get_query = SQL(
            """
WITH
{to_summarize} AS (
	SELECT
		{experiment_id},
		{experiment_column}
	FROM
		{experiment}
	WHERE TRUE
        AND EXISTS (
            SELECT 1
            FROM {run_table}
            WHERE TRUE
                AND {run_table}.{status} = {status_complete}
                AND {run_table}.{experiment_id} IS NOT NULL
            )
	LIMIT {experiment_limit}
	FOR UPDATE SKIP LOCKED
),
updated_runs AS (
UPDATE {run_table} SET
	{status} = {status_summarized}
FROM
	{to_summarize}
WHERE TRUE
	AND {run_table}.{status} = {status_complete}
    AND {run_table}.{experiment_id} IS NOT NULL
	AND {run_table}.{experiment_id} = {to_summarize}.{experiment_id}
)
SELECT
	{to_summarize}.{experiment_id},
	{to_summarize}.{experiment_column},
	{run_table}.{id},
	{run_table}.{history_column}
FROM
	{to_summarize}
	INNER JOIN {run_table} ON ({run_table}.{experiment_id} = {to_summarize}.{experiment_id})
ORDER BY {experiment_id}
;"""
        ).format(**format_args)

        rows = []
        with ConnectionManager(database.credentials) as connection:
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
                return row[result_columns[column]]

            this_experiment_id = value_of(run_table.experiment_id)
            if this_experiment_id != experiment_id:
                summarize_experiment()
                encoded_histories.clear()
                experiment_id = this_experiment_id
                marshaled_experiment = value_of(experiment_table.experiment)

            encoded_histories.append(
                (value_of(run_table.id), value_of(run_table.history))
            )
        summarize_experiment()

        try:
            database.store_summaries(summaries)
        except Exception as e:
            result.num_experiments_excepted += len(summaries)
            result.num_experiments_updated -= len(summaries)
            print(f"Error storing summaries: {e}")
            import traceback

            traceback.print_exc()

        return result
