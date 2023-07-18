from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable
from uuid import UUID, uuid4


from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class WorkerTaskContext:
    from dmp.worker import Worker
    from dmp.task.task import Task
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

    def checkpoint_task(
        self,
        task: TrainingExperiment,
        model: ModelInfo,
        experiment_history: Dict[str, Any],
    ) -> TrainingExperimentCheckpoint:
        # + save checkpoint
        #     + to disk
        #     + to db
        task.resume_from = self.save_model(
            model, task.get_current_epoch(experiment_history)
        )

        # + save history to db
        self.worker._result_logger.log()

        # + update Job & Task to resume on failure
        # + add table to track job/task execution history?

        return task.resume_from

    def save_model(
        self,
        model: ModelInfo,
        epoch: TrainingEpoch,
    ) -> TrainingExperimentCheckpoint:
        from psycopg.sql import SQL, Composed, Identifier
        from jobqueue.connection_manager import ConnectionManager
        import dmp.keras_interface.model_serialization as model_serialization

        model_path = model_serialization.get_path_for_model_savepoint(
            self.id,
            epoch.model_number,
            epoch.model_epoch,
        )

        model_serialization.save_model_data(self.task, model, model_path)

        if self.schema is not None:
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
                        epoch.model_number,
                        epoch.model_epoch,
                        epoch.epoch,
                    ),
                    binary=True,
                )

        return TrainingExperimentCheckpoint(
            self.id,
            True,
            True,
            epoch,
        )


from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
