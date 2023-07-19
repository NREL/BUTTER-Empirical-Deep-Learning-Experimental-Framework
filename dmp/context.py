from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple
from uuid import UUID, uuid4

from typing import TYPE_CHECKING

from jobqueue.connection_manager import ConnectionManager
import pandas
from psycopg.sql import SQL, Identifier
from dmp.parquet_util import make_dataframe_from_dict

from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.task.run import Run
from dmp.uuid_tools import object_to_uuid

if TYPE_CHECKING:
    from dmp.task.experiment.experiment import Experiment
    from dmp.task.experiment.training_experiment.run_spec import RunSpec
    from dmp.task.experiment.training_experiment.training_experiment import (
        TrainingExperiment,
    )
    from dmp.worker import Worker
    from dmp.task.task import Task
    from dmp.model.model_info import ModelInfo
    from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
    from jobqueue.job import Job
    from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
    from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
        TrainingExperimentCheckpoint,
    )


@dataclass
class Context:
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

    @property
    def run(self) -> Run:
        if not isinstance(self.task, Run):
            raise TypeError()
        return self.task

    @property
    def experiment(self) -> Experiment:
        return self.run.experiment

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

    def get_experiment_id(self) -> UUID:
        return object_to_uuid(self.experiment)

    def record_history(
        self,
        history: pandas.DataFrame,
        extended_history: pandas.DataFrame,
    ) -> None:
        if self.schema is not None:
            self.schema.record_history(
                [
                    (
                        self.job.id,
                        self.get_experiment_id(),
                        history,
                        extended_history,
                    )
                ]
            )

    def update_summary(
        self,
    ) -> None:
        if self.schema is not None:
            experiment_id = self.get_experiment_id()
            summary = self.experiment.summarize(
                self.schema.get_experiment_run_histories(experiment_id)
            )
            self.schema.store_summary(self.run.experiment, experiment_id, summary)  # type: ignore

    def save_model(
        self,
        model: ModelInfo,
        epoch: TrainingEpoch,
    ) -> TrainingExperimentCheckpoint:
        import dmp.keras_interface.model_serialization as model_serialization

        model_path = model_serialization.get_path_for_model_savepoint(
            self.id,
            epoch.model_number,
            epoch.model_epoch,
        )

        model_serialization.save_model_data(self.run, model, model_path)

        if self.schema is not None:
            self.schema.save_model(self.id, epoch)

        return TrainingExperimentCheckpoint(
            self.id,
            True,
            True,
            epoch,
        )

    # def checkpoint_task(
    #     self,
    #     task: TrainingExperiment,
    #     model: ModelInfo,
    #     experiment_history: Dict[str, Any],
    # ) -> TrainingExperimentCheckpoint:
    #     # + save checkpoint
    #     #     + to disk
    #     #     + to db
    #     task.resume_from = self.save_model(
    #         model, task.get_current_epoch(experiment_history)
    #     )

    #     # + save history to db
    #     self.worker._result_logger.log()

    #     # + update Job & Task to resume on failure
    #     # + add table to track job/task execution history?

    #     return task.resume_from
