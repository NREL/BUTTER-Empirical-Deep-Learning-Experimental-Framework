from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from typing import TYPE_CHECKING

import pandas

from dmp.parquet_util import make_dataframe_from_dict

from dmp.run_entry import RunEntry
from dmp.task.run_status import RunStatus

from dmp.uuid_tools import object_to_uuid

if TYPE_CHECKING:
    from dmp.script.worker import Worker
    from dmp.task.experiment.experiment import Experiment
    from dmp.task.experiment.training_experiment.run_spec import RunConfig
    from dmp.task.run import Run
    from dmp.task.run_command import RunCommand
    from dmp.model.model_info import ModelInfo
    from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
    from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
    from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
        TrainingExperimentCheckpoint,
    )


@dataclass
class Context:
    worker: Worker
    run_entry: RunEntry

    @property
    def database(self) -> PostgresInterface:
        return self.worker.database

    @property
    def id(self) -> UUID:
        return self.run_entry.id

    @property
    def info(self) -> Dict[str, Any]:
        return self.worker.info

    @property
    def run(self) -> Run:
        from dmp.task.run import Run

        if not isinstance(self.run_entry.command, Run):
            raise TypeError()
        return self.run_entry.command

    @property
    def experiment(self) -> Experiment:
        return self.run.experiment

    def push_runs(self, runs: Iterable[Run]) -> None:
        self.database.push_runs(
            [
                RunEntry(
                    queue=self.run_entry.queue,
                    status=RunStatus.Queued,
                    priority=self.run_entry.priority + 1,
                    id=uuid4(),
                    start_time=None,
                    update_time=None,
                    worker_id=None,
                    parent_id=self.id,
                    experiment_id=None,
                    command=run,
                    history=None,
                    extended_history=None,
                    error_message=None,
                )
                for run in runs
            ]
        )

    def get_experiment_id(self) -> UUID:
        return object_to_uuid(self.experiment)

    def update_run(self) -> None:
        self.run_entry.experiment_id = self.get_experiment_id()
        print(
            f"unpdating run {self.run_entry.id} with experiment id {self.run_entry.experiment_id}..."
        )
        self.database.update_runs((self.run_entry,))

    def get_run_history(
        self,
        run_id: Optional[UUID],
        epoch: TrainingEpoch,
    ) -> Dict[str, List]:
        if run_id is None:
            return {}

        if run_id == self.run_entry.id:
            history_df = self.run_entry.history
            extended_history_df = self.run_entry.extended_history
        else:
            history_df, extended_history_df = self.database.get_run_history(run_id)

        if history_df is None:
            return {}

        history_df.sort_values(["fit_number", "fit_epoch", "epoch"], inplace=True)

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

        history_df = history_df[history_df["epoch"] <= epoch.epoch]
        return history_df.to_dict(orient="list")  # type: ignore

    def update_summary(
        self,
    ) -> None:
        if self.database is not None:
            experiment_id = self.get_experiment_id()
            # print(f"loading summaries for experiment {experiment_id}...")
            # print(f"experiment marshalling: \n")
            # import simplejson
            # from dmp.marshaling import marshal

            # print(
            #     simplejson.dumps(
            #         marshal.marshal(self.experiment), sort_keys=True, indent="  "
            #     )
            # )

            summary = self.experiment.summarize(
                self.database.get_run_histories_for_experiment(experiment_id)
            )
            if summary is not None:
                self.database.store_summary(experiment_id, summary)  # type: ignore

    def save_model(
        self,
        model: ModelInfo,
        epoch: TrainingEpoch,
    ) -> TrainingExperimentCheckpoint:
        import dmp.keras_interface.model_serialization as model_serialization
        from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
            TrainingExperimentCheckpoint,
        )

        # print(
        #     f"\n\n\n========== saving model data run:{self.run} model_path:{model_path} model: {model} ==========\n\n\n"
        # )

        model_serialization.save_model_data(
            self.id,
            model,
            epoch,
            self.run_entry.parent_id,
        )

        # if self.schema is not None:
        #     self.schema.save_model(self.id, epoch)

        return TrainingExperimentCheckpoint(
            self.id,
            True,
            True,
            replace(epoch),
        )
