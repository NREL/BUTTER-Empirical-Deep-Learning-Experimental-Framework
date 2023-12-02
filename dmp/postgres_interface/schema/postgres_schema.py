from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union

from uuid import UUID
from jobqueue.connection_manager import ConnectionManager
from jobqueue.job_status import JobStatus
import pandas
from psycopg import ClientCursor

# import psycopg
from psycopg.sql import Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb
import pyarrow.parquet
from dmp.model.model_info import ModelInfo
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder

from dmp.parquet_util import (
    convert_bytes_to_dataframe,
    convert_dataframe_to_bytes,
)
from dmp.postgres_interface.element.column_group import ColumnGroup

from dmp.postgres_interface.postgres_interface_common import json_dump_function
from dmp.postgres_interface.schema.experiment_table import ExperimentTable
from dmp.postgres_interface.schema.checkpoint_table import CheckpointTable
from dmp.postgres_interface.schema.history_table import HistoryTable
from dmp.postgres_interface.schema.run_data_table import RunDataTable
from dmp.postgres_interface.schema.run_status_table import RunStatusTable
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)
from dmp.task.run import Run

from psycopg.types.json import set_json_dumps

set_json_dumps(json_dump_function)


class PostgresSchema:
    credentials: Dict[str, Any]

    experiment_id_column: str

    run_status: RunStatusTable = RunStatusTable()
    run_data: RunDataTable = RunDataTable()
    history: HistoryTable = HistoryTable()
    experiment: ExperimentTable = ExperimentTable()
    checkpoint: CheckpointTable = CheckpointTable()

    log_query_suffix: Composed

    def __init__(
        self,
        credentials: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.credentials = credentials

    #     def update_job(self, job, run):
    #         from dmp.marshaling import marshal
    #         query = SQL(
    # """
    # UPDATE {job_data_table} SET
    #     {command} = {command_value}
    #     WHERE
    #     {job_id} = {job_id_value}
    # ;"""
    #         ).format(
    #             history_table=history_table.identifier,
    #             input_colums=input_colums.columns_sql,
    #             casting_clause=input_colums.casting_sql,
    #             input_placeholders=input_colums.placeholders_for_values(
    #                 len(results)
    #             ),
    #             input_table=Identifier("input_table"),
    #         )
    #         print(query)

    #         with ConnectionManager(self.credentials) as connection:
    #             connection.execute(
    #                 query,
    #                 prepared_results,
    #                 binary=True,
    #             )

    def record_history(
        self,
        results: Sequence[Tuple[UUID, UUID, pandas.DataFrame, pandas.DataFrame]],
    ) -> None:
        # prepare histories:
        prepared_results = list(
            chain(
                *(
                    (
                        run_id,
                        experiment_id,
                        convert_dataframe_to_bytes(history),
                        convert_dataframe_to_bytes(extended_history),
                    )
                    for run_id, experiment_id, history, extended_history in results
                )
            )
        )

        history_table = self.history
        input_colums = ColumnGroup(
            history_table.id,
            history_table.experiment_id,
            history_table.history,
            history_table.extended_history,
        )

        query = SQL(
            """
                INSERT INTO {history_table} ( {input_colums} )
                SELECT
                {casting_clause}
                FROM
                ( VALUES {input_placeholders} ) AS {input_table} ({input_colums})
                ON CONFLICT ({id}) DO UPDATE SET {update_clause}
                ;"""
        ).format(
            history_table=history_table.identifier,
            input_colums=input_colums.columns_sql,
            casting_clause=input_colums.casting_sql,
            input_placeholders=input_colums.placeholders_for_values(len(results)),
            input_table=Identifier("_input_table"),
            id=history_table.id.identifier,
            update_clause=sql_comma.join(
                (
                    SQL("{column} = EXCLUDED.{column}").format(column=column.identifier)
                    for column in (
                        history_table.experiment_id,
                        history_table.history,
                        history_table.extended_history,
                    )
                )
            ),
        )
        # print(query)

        with ConnectionManager(self.credentials) as connection:
            connection.execute(
                query,
                prepared_results,
                binary=True,
            )
            # with ClientCursor(connection) as cursor:
            #     mog = cursor.mogrify(
            #         query,
            #         prepared_results,
            #     )
            #     print(f"history query: {mog}")

    def get_run_history(
        self,
        run_id: UUID,
    ) -> Optional[pandas.DataFrame]:
        history_table = self.history
        columns = ColumnGroup(
            history_table.history,
            history_table.extended_history,
        )
        query = SQL(
            """
SELECT
    {columns}
FROM
    {history_table}
WHERE
    {history_table}.{id} = {run_id_value}
LIMIT 1
;"""
        ).format(
            columns=columns.columns_sql,
            history_table=history_table.identifier,
            id=history_table.id.identifier,
            run_id_value=Literal(run_id),
        )

        history_bytes = None
        extended_history_bytes = None
        with ConnectionManager(self.credentials) as connection:
            with connection.cursor(binary=True) as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
                if row is None:
                    return None
                history_bytes = row[0]
                extended_history_bytes = row[1]

        history_df = None
        if history_bytes is not None:
            history_df = convert_bytes_to_dataframe(history_bytes)
            if history_df is not None:
                history_df.sort_values(
                    ["epoch", "fit_number", "fit_epoch"], inplace=True
                )

                if extended_history_bytes is not None:
                    extended_history_df = convert_bytes_to_dataframe(
                        extended_history_bytes
                    )
                    if extended_history_df is not None:
                        merge_keys = ["epoch"]
                        if "fit_number" in extended_history_df:
                            merge_keys.extend(["fit_number", "fit_epoch"])

                        history_df = history_df.merge(
                            extended_history_df,
                            left_on=merge_keys,
                            right_on=merge_keys,
                            suffixes=(None, "_extended"),
                        )

        return history_df

    def get_experiment_run_histories(
        self,
        experiment_id: UUID,
    ) -> List[Tuple[UUID, pandas.DataFrame]]:
        history_table = self.history
        run_status_table = self.run_status
        query = SQL(
            """
SELECT
    {history_table}.{id},
    {history_table}.{history}
FROM
    {history_table}
    INNER JOIN {run_status_table} ON ({history_table}.{id} = {run_status_table}.{id})
WHERE TRUE
    AND {history_table}.{experiment_id} = {experiment_id_value}
    AND {history} IS NOT NULL
    AND {run_status_table}.{status} = {status_value}
;"""
        ).format(
            id=history_table.id.identifier,
            history=history_table.history.identifier,
            history_table=history_table.identifier,
            experiment_id=history_table.experiment_id.identifier,
            experiment_id_value=Literal(experiment_id),
            run_status_table=run_status_table.identifier,
            status=run_status_table.status.identifier,
            status_value=Literal(int(JobStatus.Complete)),
        )

        results = []
        with ConnectionManager(self.credentials) as connection:
            with connection.cursor(binary=True) as cursor:
                cursor.execute(query)
                for row in cursor.fetchall():
                    results.append((row[0], convert_bytes_to_dataframe(row[1])))
        return results

    def store_summary(
        self,
        experiment: TrainingExperiment,
        experiment_id: UUID,
        summary: ExperimentSummaryRecord,
    ) -> None:
        self.store_summaries([(experiment, experiment_id, summary)])

    def store_summaries(
        self,
        summaries: Sequence[Tuple[TrainingExperiment, UUID, ExperimentSummaryRecord]],
    ) -> None:
        from dmp.marshaling import marshal

        prepared_values = list(
            chain(
                *(
                    (
                        experiment_id,
                        Jsonb(marshal.marshal(experiment)),
                        summary.num_runs,
                        convert_dataframe_to_bytes(summary.by_epoch),
                        convert_dataframe_to_bytes(summary.by_loss),
                        convert_dataframe_to_bytes(summary.by_progress),
                        convert_dataframe_to_bytes(summary.epoch_subset),
                    )
                    for experiment, experiment_id, summary in summaries
                )
            )
        )

        experiment_table = self.experiment
        input_columns = ColumnGroup(
            experiment_table.experiment_id,
            experiment_table.experiment,
            experiment_table.num_runs,
            experiment_table.by_epoch,
            experiment_table.by_loss,
            experiment_table.by_progress,
            experiment_table.epoch_subset,
        )

        query = SQL(
            """
INSERT INTO {experiment_table} ( {input_columns} )
SELECT
{casting_clause}
FROM
( VALUES {input_placeholders} ) AS {input_table} ({input_columns})
ON CONFLICT ({experiment_id}) DO UPDATE SET {update_clause}
;"""
        ).format(
            experiment_table=experiment_table.identifier,
            input_columns=input_columns.columns_sql,
            casting_clause=input_columns.casting_sql,
            input_placeholders=input_columns.placeholders_for_values(len(summaries)),
            input_table=Identifier("_input_table"),
            experiment_id=experiment_table.experiment_id.identifier,
            update_clause=sql_comma.join(
                (
                    SQL("{column} = EXCLUDED.{column}").format(column=column.identifier)
                    for column in (
                        experiment_table.num_runs,
                        experiment_table.by_epoch,
                        experiment_table.by_loss,
                        experiment_table.by_progress,
                        experiment_table.epoch_subset,
                    )
                )
            ),
        )
        # print(query)

        with ConnectionManager(self.credentials) as connection:
            connection.execute(
                query,
                prepared_values,
                binary=True,
            )
