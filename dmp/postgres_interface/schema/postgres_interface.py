from dataclasses import dataclass
import dataclasses
from itertools import chain
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, List, Union

from uuid import UUID
from jobqueue.connection_manager import ConnectionManager
from jobqueue.cursor_manager import CursorManager
from jobqueue.job_queue import JobQueue
import pandas
from psycopg import ClientCursor

# import psycopg
from psycopg.sql import Identifier, SQL, Composed, Literal
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder

from dmp.parquet_util import (
    convert_bytes_to_dataframe,
    convert_dataframe_to_bytes,
)
from dmp.postgres_interface.element.column_group import ColumnGroup

from dmp.postgres_interface.postgres_interface_common import json_dump_function
from dmp.postgres_interface.schema.experiment_table import ExperimentTable
from dmp.postgres_interface.schema.run_table import RunTable
from dmp.run_entry import RunEntry
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord

from psycopg.types.json import set_json_dumps

from dmp.task.run_status import RunStatus

set_json_dumps(json_dump_function)


class PostgresInterface:
    _credentials: Dict[str, Any]
    _queue: int
    _run_table: RunTable = RunTable()

    _experiment_table: ExperimentTable = ExperimentTable()

    _pop_query: Composed
    _update_columns: ColumnGroup
    _run_columns: ColumnGroup
    _dataframe_columns: ColumnGroup

    def __init__(
        self,
        credentials: Dict[str, Any],
        queue: int,
    ) -> None:
        self._credentials = credentials
        self._queue = queue

        run_table = self._run_table

        self._update_columns = ColumnGroup(
            run_table.queue,
            run_table.status,
            run_table.priority,
            run_table.id,
            run_table.start_time,
            run_table.update_time,
            run_table.worker_id,
            run_table.parent_id,
            run_table.experiment_id,
            run_table.command,
            run_table.history,
            run_table.extended_history,
            run_table.error_message,
        )

        self._run_columns = ColumnGroup(run_table.id) + self._update_columns

        self._dataframe_columns = ColumnGroup(
            run_table.history, run_table.extended_history
        )

        self._pop_query = SQL(
            """
UPDATE {run_table}
    SET
        status = {claimed_status},
        worker = %b,
        start_time = NOW(),
        update_time = NOW()
    FROM p
    WHERE p.id = q.id
RETURNING
    {returning_clause}
) LIMIT %b;"""
        ).format(
            run_table=self._run_table.identifier,
            claimed_status=Literal(int(RunStatus.Claimed)),
            returning_clause=self._run_columns.columns_sql,
        )

    def pop_runs(
        self,
        worker_id: Optional[UUID] = None,
        n: int = 1,
    ) -> List[RunEntry]:
        if n <= 0:
            return []

        rows = []
        with ConnectionManager(self._credentials) as connection:
            with ClientCursor(connection) as cursor:
                cursor.execute(self._pop_query, (worker_id, n), binary=True)
                rows = cursor.fetchall()

        for i, row in enumerate(rows):
            run = self._get_run_from_row(row)
            rows[i] = run

        return rows

    def push_runs(
        self,
        runs: Sequence[RunEntry],
    ) -> None:
        run_table = self._run_table
        experiment_table = self._experiment_table
        run_columns = self._run_columns

        def make_query(num_runs_in_block: int) -> Composed:
            return SQL(
                """
WITH {input_table} AS (
    SELECT {casting_clause} FROM ( VALUES {input_placeholders} ) AS {input_table} ({input_columns})
),
{experiment_insert} AS (
    INSERT INTO {experiment_table} ( {experiment_id}, {experiment_column} )
    SELECT {experiment_id}, ({command}->{experiment_key}) {experiment_column}
    FROM {input_table}
    WHERE TRUE
        AND {experiment_id} IS NOT NULL
        AND {command} IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM {experiment_table} WHERE {experiment_id} = {input_table}.{experiment_id})
    ON CONFLICT ({experiment_id}) DO NOTHING
)
INSERT INTO {run_table} ({run_columns})
FROM {input_table}
ON CONFLICT ({id}) DO UPDATE SET
    {set_clause}
"""
            ).format(
                experiment_insert=Identifier("_experiment_insert"),
                experiment_key=Literal("experiment"),
                command=run_table.command.identifier,
                experiment_table=experiment_table.identifier,
                experiment_id=experiment_table.experiment_id.identifier,
                experiment_column=experiment_table.experiment.identifier,
                run_table=run_table.identifier,
                casting_clause=run_columns.casting_sql,
                input_table=Identifier("_input_table"),
                input_columns=run_columns.columns_sql,
                input_placeholders=run_columns.placeholders_for_values(
                    num_runs_in_block
                ),
                set_clause=self._update_columns.set_clause(Identifier("EXCLUDED")),
                id=run_table.id.identifier,
            )

        return self._edit_runs(runs, make_query)

    def update_runs(
        self,
        runs: Sequence[RunEntry],
    ) -> None:
        run_table = self._run_table
        experiment_table = self._experiment_table
        run_columns = self._run_columns

        def make_query(num_runs_in_block: int) -> Composed:
            return SQL(
                """
WITH {input_table} AS (
    SELECT {casting_clause} FROM ( VALUES {input_placeholders} ) AS {input_table} ({input_columns})
),
{experiment_insert} AS (
    INSERT INTO {experiment_table} ( {experiment_id}, {experiment_column} )
    SELECT {experiment_id}, ({command}->{experiment_key}) {experiment_column}
    FROM {input_table}
    WHERE TRUE
        AND {experiment_id} IS NOT NULL
        AND {command} IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM {experiment_table} WHERE {experiment_id} = {input_table}.{experiment_id})
    ON CONFLICT ({experiment_id}) DO NOTHING
)
UPDATE {run_table} SET
    {set_clause}
FROM {input_table}
WHERE {run_table}.{id} = {input_table}.{id}
"""
            ).format(
                experiment_insert=Identifier("_experiment_insert"),
                experiment_key=Literal("experiment"),
                command=run_table.command.identifier,
                experiment_table=experiment_table.identifier,
                experiment_id=experiment_table.experiment_id.identifier,
                experiment_column=experiment_table.experiment.identifier,
                run_table=run_table.identifier,
                set_clause=run_columns.set_clause,
                input_table=Identifier("_input_table"),
                input_columns=run_columns.columns_sql,
                casting_clause=run_columns.casting_sql,
                input_placeholders=run_columns.placeholders_for_values(
                    num_runs_in_block
                ),
                id=run_table.id.identifier,
            )

        return self._edit_runs(runs, make_query)

    def _edit_runs(
        self,
        runs: Sequence[RunEntry],
        make_query: Callable,
    ) -> None:
        if len(runs) <= 0:
            return

        prepared_values = []
        for run in runs:
            prepared_values.extend(self._get_column_values_for_run(run))

        num_cols_per_row = len(self._run_columns.columns)
        block_size = 65500 // num_cols_per_row

        i = 0
        with CursorManager(self._credentials, binary=True) as cursor:
            with cursor.connection.transaction():
                while i < len(runs):
                    num_runs_in_block = max(block_size, len(runs) - i)
                    query = make_query(num_runs_in_block)
                    cursor.execute(
                        query,
                        prepared_values[
                            i * num_cols_per_row : (i + block_size) * num_cols_per_row
                        ],
                        binary=True,
                    )

                    i += block_size

    def _get_run_from_row(self, row: Any) -> RunEntry:
        from dmp.marshaling import marshal

        run_columns = self._run_columns

        run = RunEntry(
            **{column.name: row[run_columns[column]] for column in run_columns.columns}
        )
        run.history = convert_bytes_to_dataframe(run.history)  # type: ignore
        run.extended_history = convert_bytes_to_dataframe(run.extended_history)  # type: ignore
        run.command = marshal.demarshal(run.command)
        if run.command is not None:
            run.command.run_entry = run
        return run

    def _get_column_values_for_run(self, run: RunEntry) -> Iterable[Any]:
        from dmp.marshaling import marshal

        run = dataclasses.replace(run)

        command = run.command
        command.run_entry = None  # type: ignore
        run.command = marshal.marshal(command)
        command.run_entry = run  # type: ignore

        run.history = convert_dataframe_to_bytes(run.history)  # type: ignore
        run.extended_history = convert_dataframe_to_bytes(run.extended_history)  # type: ignore
        run_dict = dataclasses.asdict(run)

        return (run_dict[column.name] for column in self._run_columns.columns)

    def get_run_history(
        self,
        run_id: UUID,
    ) -> Tuple[Optional[pandas.DataFrame], Optional[pandas.DataFrame]]:
        run_table = self._run_table
        columns = ColumnGroup(
            run_table.history,
            run_table.extended_history,
        )
        query = SQL(
            """
SELECT
    {columns}
FROM
    {run_table}
WHERE
    {id} = %b
LIMIT 1
;"""
        ).format(
            columns=columns.columns_sql,
            history_table=run_table.identifier,
            id=run_table.id.identifier,
            run_id=Literal(run_id),
        )

        history_bytes = None
        extended_history_bytes = None
        with ConnectionManager(self._credentials) as connection:
            with connection.cursor(binary=True) as cursor:
                cursor.execute(query, (run_id,))
                row = cursor.fetchone()
                if row is None:
                    return None, None
                history_bytes = row[0]
                extended_history_bytes = row[1]

        history_df = None
        extended_history_df = None
        if history_bytes is not None:
            history_df = convert_bytes_to_dataframe(history_bytes)
            if history_df is not None:
                history_df.sort_values(
                    ["epoch", "fit_number", "fit_epoch"], inplace=True
                )

            if extended_history_bytes is not None:
                extended_history_df = convert_bytes_to_dataframe(extended_history_bytes)
                if extended_history_df is not None:
                    merge_keys = ["epoch"]
                    if "fit_number" in extended_history_df:
                        merge_keys.extend(["fit_number", "fit_epoch"])

                    history_df = history_df.merge(  # type: ignore
                        extended_history_df,
                        left_on=merge_keys,
                        right_on=merge_keys,
                        suffixes=(None, "_extended"),
                    )

        return history_df, extended_history_df

    def get_run_histories_for_experiment(
        self,
        experiment_id: UUID,
    ) -> List[Tuple[UUID, pandas.DataFrame]]:
        run_table = self._run_table
        run_status_table = self._run_table
        query = SQL(
            """
SELECT
    {id},
    {history}
FROM
    {run_table}
WHERE TRUE
    AND {experiment_id} = %b
    AND {history} IS NOT NULL
    AND {status} >= {status_value}
;"""
        ).format(
            id=run_table.id.identifier,
            history=run_table.history.identifier,
            history_table=run_table.identifier,
            experiment_id=run_table.experiment_id.identifier,
            run_status_table=run_status_table.identifier,
            status=run_status_table.status.identifier,
            status_value=Literal(int(RunStatus.Complete)),
        )

        results = []
        with ConnectionManager(self._credentials) as connection:
            with connection.cursor(binary=True) as cursor:
                cursor.execute(query, (experiment_id,))
                for row in cursor.fetchall():
                    results.append((row[0], convert_bytes_to_dataframe(row[1])))
        return results

    def store_summary(
        self,
        experiment_id: UUID,
        summary: ExperimentSummaryRecord,
    ) -> None:
        self.store_summaries([(experiment_id, summary)])

    def store_summaries(
        self,
        summaries: Sequence[Tuple[UUID, ExperimentSummaryRecord]],
    ) -> None:
        from dmp.marshaling import marshal

        if len(summaries) == 0:
            return

        prepared_values = list(
            chain(
                *(
                    (
                        experiment_id,
                        summary.num_runs,
                        convert_dataframe_to_bytes(summary.by_epoch),
                        convert_dataframe_to_bytes(summary.by_loss),
                    )
                    for experiment_id, summary in summaries
                )
            )
        )

        experiment_table = self._experiment_table
        input_columns = ColumnGroup(
            experiment_table.experiment_id,
            experiment_table.num_runs,
            experiment_table.by_epoch,
            experiment_table.by_loss,
        )

        query = SQL(
            """
UPDATE {experiment_table} SET
    {num_runs} = {input_table}.{num_runs},
    {by_epoch} = {input_table}.{by_epoch},
    {by_loss} = {input_table}.{by_loss}
FROM
    (SELECT {casting_clause} FROM ( VALUES {input_placeholders} ) AS {input_table} ({input_columns})) {input_table}
WHERE
    {experiment_table}.{experiment_id} = {input_table}.{experiment_id}
;"""
        ).format(
            experiment_table=experiment_table.identifier,
            num_runs=experiment_table.num_runs.identifier,
            by_epoch=experiment_table.by_epoch.identifier,
            by_loss=experiment_table.by_loss.identifier,
            input_columns=input_columns.columns_sql,
            casting_clause=input_columns.casting_sql,
            input_placeholders=input_columns.placeholders_for_values(len(summaries)),
            input_table=Identifier("_input_table"),
            experiment_id=experiment_table.experiment_id.identifier,
        )
        # print(query)

        with ConnectionManager(self._credentials) as connection:
            connection.execute(
                query,
                prepared_values,
                binary=True,
            )
