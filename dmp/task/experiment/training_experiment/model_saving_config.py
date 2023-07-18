from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Set

from jobqueue.job import Job
from dmp.model.model_info import ModelInfo
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.training_experiment.epoch import TrainingEpoch
from dmp.worker import Worker
from dmp.worker_task_context import WorkerTaskContext


@dataclass
class ModelSavingConfig:
    """
    Saves every fixed_interval steps, up to the fixed_threshold step, and then saves every exponential_rate ^ i steps, where i is a positive integer and exponential_rate ^ i >= fixed_threshold.
    To only save initial and/or final models, set fixed_threshold = 0, and exponential_rate = 0.0.
    To disable fixed savepoint spacing set fixed_threshold = 0.
    To disable exponential savepoint spacing set exponential_rate = 0.0.
    """

    save_initial_model: bool
    save_trained_model: bool

    save_epochs: List[int]  # specific global epochs to save at
    save_model_epochs: List[int]

    fixed_interval: int
    fixed_threshold: int
    exponential_rate: float

    # specific model epochs to save at for every model

    def make_save_model_callback(
        self,
        context: WorkerTaskContext,
        epoch: TrainingEpoch,
        model_info: ModelInfo,
        # checkpoint_interval: float,
    ):
        import time
        from dmp.task.experiment.training_experiment.model_saving_callback import (
            ModelSavingCallback,
        )

        class SaveCallback(ModelSavingCallback):
            def __init__(self, parent: ModelSavingConfig):
                super().__init__()
                self.parent: ModelSavingConfig = parent
                self.save_model_epochs: Set[int] = set(parent.save_model_epochs)
                self.save_epochs: Set[int] = set(parent.save_epochs)

                self.epoch: TrainingEpoch = dataclass.replace(epoch)
                self.epoch.model_number -= 1

                self.last_saved_epoch: TrainingEpoch = dataclass.replace(self.epoch)
                self.last_saved_epoch.epoch -= 1
                self.last_saved_epoch.model_epoch -= 1

                self.last_save_time: float = time.time()

                # self.history : Dict[str, List] = {}
                self._saved_epochs: List[TrainingEpoch] = []

            @property
            def saved_epochs(self) -> List[TrainingEpoch]:
                return self._saved_epochs

            def on_train_begin(self, logs=None) -> None:
                self.epoch.count_new_model()
                if self.parent.save_initial_model:
                    self.save_model()

            def on_epoch_end(self, epoch, logs=None) -> None:
                self.epoch.count_new_epoch()

                model_epoch = self.epoch.model_epoch
                parent = self.parent
                if (
                    self.epoch.epoch in self.save_epochs
                    or model_epoch in self.save_model_epochs
                ):
                    # specified epoch
                    pass
                elif (
                    parent.fixed_threshold > 0 and model_epoch <= parent.fixed_threshold
                ):
                    # fixed regime
                    if model_epoch % parent.fixed_interval != 0:
                        return
                elif parent.exponential_rate <= 0.0:
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
                if self.last_saved_epoch == self.epoch:
                    return

                self.last_saved_epoch = dataclass.replace(self.epoch)
                self._saved_epochs.append(dataclass.replace(self.epoch))
                context.save_model(model_info, self.last_saved_epoch)

        return SaveCallback(self)
