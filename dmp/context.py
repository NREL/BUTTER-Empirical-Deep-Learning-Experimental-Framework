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

from dmp.uuid_tools import object_to_uuid

if TYPE_CHECKING:
    from dmp.task.experiment.experiment import Experiment
    from dmp.task.experiment.training_experiment.run_spec import RunSpec
    from dmp.task.run import Run
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
        from dmp.task.run import Run

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
            [
                Job(
                    parent=self.id,
                    priority=self.job.priority,
                    command=marshal.marshal(task),
                )
                for task in tasks
            ]
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
        print(f"record_history\n{history}\nextended:\n{extended_history}")

        if self.schema is not None:
            experiment_id = self.get_experiment_id()
            print(f"storing experiment {experiment_id} history...")
            self.schema.record_history(
                [
                    (
                        self.job.id,
                        experiment_id,
                        history,
                        extended_history,
                    )
                ]
            )
            print(
                f"*******************************************************\nstored run\n'{self.id}'\n experiment'{self.get_experiment_id()}' history..."
            )
            print(f"experiment marshalling: \n")
            import simplejson
            from dmp.marshaling import marshal

            print(
                simplejson.dumps(
                    marshal.marshal(self.experiment), sort_keys=True, indent="  "
                )
            )

    def update_task(self) -> None:
        if self.worker is None or self.worker.job_queue is None:
            return

        from dmp.marshaling import marshal

        self.job.command = marshal.marshal(self.task)
        self.worker.job_queue.update_job_command(self.job)

    def update_summary(
        self,
    ) -> None:
        if self.schema is not None:
            experiment_id = self.get_experiment_id()
            print(f"loading summaries for experiment {experiment_id}...")
            print(f"experiment marshalling: \n")
            import simplejson
            from dmp.marshaling import marshal

            print(
                simplejson.dumps(
                    marshal.marshal(self.experiment), sort_keys=True, indent="  "
                )
            )

            summary = self.experiment.summarize(
                self.schema.get_experiment_run_histories(experiment_id)
            )
            if summary is not None:
                self.schema.store_summary(self.run.experiment, experiment_id, summary)  # type: ignore

    def save_model(
        self,
        model: ModelInfo,
        epoch: TrainingEpoch,
    ) -> TrainingExperimentCheckpoint:
        import dmp.keras_interface.model_serialization as model_serialization
        from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
            TrainingExperimentCheckpoint,
        )

        model_path = model_serialization.get_path_for_model_savepoint(
            self.id,
            epoch.model_number,
            epoch.model_epoch,
        )

        print(
            f"\n\n\n========== saving model data run:{self.run} model_path:{model_path} model: {model} ==========\n\n\n"
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
