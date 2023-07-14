from dataclasses import dataclass
from typing import Any, Dict, Iterable
from uuid import UUID, uuid4


@dataclass
class WorkerTaskContext:
    from dmp.worker import Worker
    from dmp.task.task import Task
    from dmp.task.experiment.a_experiment_task import AExperimentTask
    from dmp.model.model_info import ModelInfo
    from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
    from jobqueue.job import Job
    from dmp.postgres_interface.schema.postgres_schema import PostgresSchema

    worker: Worker
    job: Job
    task: Task

    @property
    def schema(self) -> PostgresSchema:
        return self.worker.schema

    @property
    def id(self) -> UUID:
        return self.job.id

    @property
    def info(self) -> Dict[str, Any]:
        return self.worker.info

    def push_tasks(self, tasks: Iterable[Task]) -> None:
        from jobqueue.job import Job
        from dmp.marshaling import marshal

        self.worker.job_queue.push(
            (
                Job(
                    parent=self.id,
                    priority=self.job.priority,
                    command=marshal.marshal(task),
                )
                for task in tasks
            )
        )

    def push_task(self, task: Task) -> None:
        self.push_tasks((task,))

    def save_model(
        self,
        model: ModelInfo,
        training_epoch: TrainingEpoch,
    ) -> None:
        from psycopg.sql import SQL, Composed, Identifier
        from jobqueue.connection_manager import ConnectionManager
        import dmp.keras_interface.model_serialization as model_serialization

        model_path = model_serialization.get_path_for_model_savepoint(
            self.id,
            training_epoch.model_number,
            training_epoch.model_epoch,
        )

        model_serialization.save_model_data(self.task, model, model_path)

        if self.schema is None:
            return

        with ConnectionManager(self.schema.credentials) as connection:
            model_table = self.schema.model
            columns = model_table.columns
            query = SQL(
                """
INSERT INTO {model_table} ( {insert_columns} )
SELECT
{casting_clause}
FROM
( VALUES ({placeholders}) ) AS V ({insert_columns})
ON CONFLICT DO NOTHING;
"""
            ).format(
                model_table=model_table.identifier,
                insert_columns=model_table.columns.columns_sql,
                casting_clause=columns.casting_sql,
                placeholders=columns.placeholders,
            )
            print(query)
            connection.execute(
                query,
                (
                    self.id,
                    training_epoch.model_number,
                    training_epoch.model_epoch,
                    training_epoch.epoch,
                ),
                binary=True,
            )
