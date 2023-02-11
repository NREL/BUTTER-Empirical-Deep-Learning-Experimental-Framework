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
        experiment_limit = 1024

        schema = worker._schema
        experiment = schema.experiment
        run = schema.run
        summary = schema.experiment_summary

        # selection_columns = ColumnGroup(
        #     experiment.experiment_attrs,
        #     experiment.experiment_properties,
        # )

        #                             AND experiment_properties @> array[2568] and experiment_attrs @> array[75, 224]
        selection = Identifier('_selection')
        # last_updated = Identifier('last_updated')

        experiment_columns = ColumnGroup(
            experiment.experiment_id,
            experiment.experiment_attrs,
            experiment.experiment_properties,
        )

        run_columns = ColumnGroup(
            run.run_timestamp,
            run.values,
            run.run_data,
            run.run_history,
        )

        result_columns = ColumnGroup(
            # bound_columns,
            experiment_columns,
            run_columns,
        )

        summary_columns = ColumnGroup(
            # summary.last_updated,
            summary.experiment_id,
            summary.most_recent_run,
            summary.by_epoch,
            summary.by_loss,
            summary.by_progress,
            summary.epoch_subset,
        )
        _values = Identifier('_values')

        format_args = dict(
            experiment_selection=experiment_columns.of(experiment.identifier),
            selection_selection=experiment_columns.of(selection),
            run_selection=run_columns.of(run.identifier),
            experiment=experiment.identifier,
            run_timestamp=run.run_timestamp.identifier,
            last_updated=summary.last_updated.identifier,
            experiment_id=experiment.experiment_id.identifier,
            run=run.identifier,
            experiment_summary=summary.identifier,
            run_id=run.run_id.identifier,
            _selection=selection,
            summary_columns=summary_columns.columns_sql,
            update_clause=sql_comma.join(
                (SQL('{c}=EXCLUDED.{c}').format(c=c)
                 for c in summary_columns.identifiers)),
            experiment_limit=Literal(experiment_limit),
            most_recent_run=summary.most_recent_run.identifier,
            summary_exists=Identifier('summary_exists'),
            experiment_locked=Identifier('_experiment_locked'),
            summary_locked=Identifier('_summary_locked'),
            _claim=Identifier('_claim'),
            _updated=Identifier('_updated'),
        )

        claim_and_get_query = SQL("""
WITH {_selection} AS 
(
    SELECT DISTINCT ON ({_selection}.{experiment_id}) 
        *
    FROM
    (
        SELECT 
            {_selection}.{run_timestamp},
            {experiment_selection}
        FROM
        (
            SELECT
                {run}.{experiment_id},
                {run}.{run_timestamp}
            FROM
                {run}
            WHERE
                NOT EXISTS (
                    SELECT
                        1
                    FROM
                        {experiment_summary}
                    WHERE
                        {experiment_summary}.{experiment_id} = {run}.{experiment_id}
                        AND {experiment_summary}.{most_recent_run} >= {run}.{run_timestamp}
                )
            ORDER BY 
                {run}.{run_timestamp} DESC, {run}.{experiment_id}
        ) {_selection}
        CROSS JOIN LATERAL
        (
            SELECT 
                {experiment_selection}
            FROM 
                {experiment}
            WHERE
                {experiment}.{experiment_id} = {_selection}.{experiment_id}
            FOR UPDATE SKIP LOCKED
        ) {experiment}
        LIMIT {experiment_limit}
    ) {_selection}
),
{_claim} AS (
    INSERT INTO {experiment_summary}
    (
        {experiment_id},
        {most_recent_run},
        {last_updated}
    )
    SELECT 
        {_selection}.{experiment_id} {experiment_id},
        {_selection}.{run_timestamp} {most_recent_run},
        CURRENT_TIMESTAMP
    FROM 
        {_selection}
    ON CONFLICT ({experiment_id}) DO UPDATE SET
        {most_recent_run} = EXCLUDED.{most_recent_run},
        {last_updated} = CURRENT_TIMESTAMP
)
SELECT
    {selection_selection},
    {run_selection}
FROM
    {_selection}
    INNER JOIN {run} ON
    (
        {run}.{experiment_id} = {_selection}.{experiment_id}
    )
ORDER BY {_selection}.{experiment_id}
;""").format(**format_args)

        def make_update_progress_query(num_summaries: int) -> Composed:
            return SQL("""
UPDATE {experiment_summary} SET
    {set_clause},
    {last_updated} = NULL
FROM 
    (
        SELECT {summary_casting_clause}
        FROM (VALUES {summary_column_placeholders} ) AS {_values} ({summary_columns})
    ) {_values}
WHERE
    {_values}.{experiment_id} = {experiment_summary}.{experiment_id}
    AND {experiment_summary}.{most_recent_run} <= {_values}.{most_recent_run}
;""").format(
                _values=_values,
                summary_casting_clause=summary_columns.casting_sql,
                set_clause=sql_comma.join([SQL('{c} = {_values}.{c}').format(
                    _values=_values,
                    c=c,
                ) for c in summary_columns.identifiers]),
                summary_column_placeholders=sql_comma.join(
                    [SQL('({})').format(summary_columns.placeholders)] *
                    num_summaries),
                **format_args,
            )

        
        rows = []
        with ConnectionManager(schema.credentials) as connection:
            # with ClientCursor(connection) as cursor:
            #     print(cursor.mogrify(claim_and_get_query))
            with connection.cursor(binary=True) as cursor:
                cursor.execute(claim_and_get_query, binary=True)
                rows = cursor.fetchall()

        summary_rows = []
        experiment_id = None
        runs = []
        most_recent_run = None
        experiment_attrs = {}
        experiment_properties = {}

        def make_summary():
            nonlocal most_recent_run, experiment_id, runs, summary_rows
            if len(runs) > 0:
                summary_rows.append((
                    experiment_id,
                    most_recent_run,
                    *self._summary_to_bytes(
                        schema, self._compute_summary(runs)),
                ))
            runs.clear()
            
        for row in rows:
            def value_of(column: Column) -> Any:
                return row[result_columns[column]]

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
            with ConnectionManager(schema.credentials) as connection:
                # with ClientCursor(connection) as cursor:
                #     print(cursor.mogrify(make_update_progress_query(num_summaries)))
                connection.execute(
                    make_update_progress_query(num_summaries),
                    list(
                        chain(*((
                            # last_updated,
                            *summary_cols,
                        ) for summary_cols in summary_rows))))

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


'''
SELECT
    *
FROM
    (
        SELECT DISTINCT experiment_id
        FROM
        (
            SELECT experiment.experiment_id
            FROM
            (
                SELECT run.experiment_id, run.run_timestamp FROM 
                run
                WHERE NOT EXISTS 
                (
                    SELECT (1) FROM "experiment_summary" 
                    WHERE
                        "experiment_summary"."most_recent_run" >= "run"."run_timestamp" 
                        AND "experiment_summary"."experiment_id" = "run"."experiment_id"
                )
            ) "selection"
            CROSS JOIN LATERAL 
            (
                SELECT
                    *
                FROM
                    "experiment"
                WHERE
                    "experiment"."experiment_id" = "selection"."experiment_id"
                FOR UPDATE SKIP LOCKED
            ) "experiment"
            WHERE "experiment".experiment_id IS NOT NULL
            LIMIT 64
        ) "_selection"
    ) "_selection"
    CROSS JOIN LATERAL (SELECT * FROM "run" WHERE "run"."experiment_id" = "_selection"."experiment_id") run
    ;
'''

'''
SELECT
    "_selection"."last_updated","_selection"."last_experiment_id",
    "_selection"."experiment_id","_selection"."experiment_attrs","_selection"."experiment_properties",
    "run"."run_timestamp","run"."run_id","run"."job_id","run"."seed","run"."slurm_job_id","run"."task_version","run"."num_nodes","run"."num_cpus","run"."num_gpus","run"."gpu_memory","run"."host_name","run"."batch","run"."run_data","run"."run_history"
FROM
    (
        SELECT 
            "_bound".*,
            "experiment".*
        FROM 
            (
                SELECT "run_timestamp" "last_updated", "experiment_id" "last_experiment_id"
                FROM "run" 
                WHERE "run_timestamp" >= 
                (
                    SELECT COALESCE(MAX("last_updated"), '1960-01-01'::timestamp) 
                    FROM "experiment_summary"
                )
                AND 
NOT EXISTS
(
    SELECT 1 
    FROM "experiment_summary"
    WHERE 
        "experiment_summary"."experiment_id" = "run"."experiment_id"
        AND "experiment_summary"."most_recent_run" >= "run"."run_timestamp"
)
                
ORDER BY "run_timestamp" ASC, "experiment_id" ASC, "run_id" ASC
                LIMIT 1
            ) "_bound"
            INNER JOIN "run" ON 
            (
                "run"."run_timestamp" >= "_bound"."last_updated" 
                AND "run"."experiment_id" >= "_bound"."last_experiment_id" 
                AND 
NOT EXISTS
(
    SELECT 1 
    FROM "experiment_summary"
    WHERE 
        "experiment_summary"."experiment_id" = "run"."experiment_id"
        AND "experiment_summary"."most_recent_run" >= "run"."run_timestamp"
)
                AND NOT EXISTS 
                (
                    SELECT 1 
                    FROM "run" "_run_backselect"
                    WHERE 
                        "_run_backselect"."experiment_id" = "run"."experiment_id" 
                        AND "_run_backselect"."run_timestamp" <= "run"."run_timestamp" 
                        AND "_run_backselect"."run_timestamp" >= "_bound"."last_updated" 
                        AND "_run_backselect"."run_id" < "run"."run_id"
                )
            )
            CROSS JOIN LATERAL 
            (
                SELECT
                    *
                FROM
                    "experiment"
                WHERE
                    "experiment"."experiment_id" = "run"."experiment_id"
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            ) "experiment"
        WHERE "experiment"."experiment_id" IS NOT NULL
        
ORDER BY "run_timestamp" ASC, "experiment_id" ASC, "run_id" ASC
        LIMIT 4
    ) "_selection"
    LEFT JOIN "run" ON ("run"."experiment_id" = "_selection"."experiment_id")
    ORDER BY "_selection"."experiment_id";
'''