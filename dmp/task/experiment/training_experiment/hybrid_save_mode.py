from dataclasses import dataclass
import math
from typing import Any, Optional

from jobqueue.job import Job
from dmp.model.model_info import ModelInfo
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.worker import Worker
from dmp.task.experiment.training_experiment.save_mode import SaveMode


@dataclass
class HybridSaveMode(SaveMode):
    '''
    Saves every fixed_interval steps, up to the fixed_threshold step, and then saves every exponential_rate ^ i steps, where i is a positive integer and exponential_rate ^ i >= fixed_threshold.
    To only save initial and/or final models, set fixed_threshold = 0, and exponential_rate = 0.0.
    To disable fixed savepoint spacing set fixed_threshold = 0.
    To disable exponential savepoint spacing set exponential_rate = 0.0.
    '''

    save_initial_model: bool
    save_trained_model: bool

    fixed_interval: int
    fixed_threshold: int
    exponential_rate: float

    def make_save_model_callback(
        self,
        worker: Worker,
        job: Job,
        task: ExperimentTask,
    ):
        import os
        import tensorflow.keras as keras
        import dmp.keras_interface.model_serialization as model_serialization
        from jobqueue.connection_manager import ConnectionManager
        from dmp.task.experiment.training_experiment.model_saving_callback import (
            ModelSavingCallback,
        )

        class SaveCallback(ModelSavingCallback):
            def __init__(self, parent: HybridSaveMode):
                super().__init__()
                self.parent: HybridSaveMode = parent
                self.model_number: int = -1
                self.epoch: int = 0
                self.model_epoch: int = 0
                self.last_saved_epoch: int = self.epoch - 1
                self.last_saved_model_number: int = self.model_number
                self.task: ExperimentTask = task
                self.model_info: Optional[
                    ModelInfo
                ] = None  # NB: must be set before calling to save model states

            def on_train_begin(self, logs=None) -> None:
                self.model_number += 1
                self.model_epoch = 0
                if self.parent.save_initial_model:
                    self.save_model()

            def on_epoch_end(self, epoch, logs=None) -> None:
                self.model_epoch += 1
                self.epoch += 1

                parent = self.parent
                model_epoch = self.model_epoch
                if parent.fixed_threshold > 0 and model_epoch <= parent.fixed_threshold:
                    # fixed regime
                    if model_epoch % parent.fixed_interval != 0:
                        return
                elif parent.exponential_rate == 0.0:
                    # exponential regime disabled
                    return
                else:
                    # exponential regime
                    denom = math.log(self.exponential_rate)
                    ratio = math.ceil(math.log(model_epoch) / denom)
                    next_ratio = math.ceil(math.log(model_epoch + 1) / denom)
                    if ratio != next_ratio:
                        return

                self.save_model()

            def on_train_end(self, logs=None) -> None:
                if self.parent.save_trained_model:
                    self.save_model()

            def save_model(self) -> None:
                model_info = self.model_info
                if model_info is None or (
                    self.last_saved_epoch == self.epoch
                    and self.last_saved_model_number == self.model_number
                ):
                    return

                self.last_saved_epoch = self.epoch
                self.last_saved_model_number = self.model_number

                model_path = model_serialization.get_path_for_model_savepoint(
                    job.id,
                    self.model_number,
                    self.model_epoch,
                )

                model_serialization.save_model_data(self.task, model_info, model_path)

                if worker.schema is None:
                    return

                from psycopg.sql import SQL, Composed, Identifier

                with ConnectionManager(worker.schema.credentials) as connection:
                    model_table = worker.schema.model
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
                        (job.id, self.model_number, self.model_epoch, self.epoch),
                        binary=True,
                    )

        return SaveCallback(self)
