from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Set

from jobqueue.job import Job
from dmp.model.model_info import ModelInfo
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.worker import Worker
from dmp.task.experiment.training_experiment.save_mode import SaveMode
from dmp.worker_task_context import WorkerTaskContext


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

    save_epochs: List[int]  # specific global epochs to save at
    save_model_epochs: List[int]

    fixed_interval: int
    fixed_threshold: int
    exponential_rate: float

    # specific model epochs to save at for every model

    def make_save_model_callback(
        self,
        context: WorkerTaskContext,
        training_epoch: TrainingEpoch,
    ):
        from dmp.task.experiment.training_experiment.model_saving_callback import ModelSavingCallback

        class SaveCallback(ModelSavingCallback):
            def __init__(self, parent: HybridSaveMode):
                super().__init__()
                self.parent: HybridSaveMode = parent
                self.save_model_epochs: Set[int] = set(
                    parent.save_model_epochs)
                self.save_epochs: Set[int] = set(parent.save_epochs)

                self.training_epoch: TrainingEpoch = dataclass.replace(
                    training_epoch)
                self.training_epoch.model_number -= 1

                self.last_saved_epoch: TrainingEpoch = dataclass.replace(
                    self.training_epoch)
                self.last_saved_epoch.epoch -= 1
                self.last_saved_epoch.model_epoch -= 1

                self.model_info: Optional[
                    ModelInfo
                ] = None  # NB: must be set before calling to save model states

                # self.history : Dict[str, List] = {}
                self.saved_epochs: List[TrainingEpoch] = []

            def on_train_begin(self, logs=None) -> None:
                self.training_epoch.count_new_model()
                if self.parent.save_initial_model:
                    self.save_model()

            def on_epoch_end(self, epoch, logs=None) -> None:
                self.training_epoch.count_new_epoch()

                model_epoch = self.training_epoch.model_epoch
                parent = self.parent
                if self.training_epoch.epoch in self.save_epochs or model_epoch in self.save_model_epochs:
                    # specified epoch
                    pass
                elif parent.fixed_threshold > 0 and model_epoch <= parent.fixed_threshold:
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
                if model_info is None or self.last_saved_epoch == self.training_epoch:
                    return

                self.last_saved_epoch = dataclass.replace(self.training_epoch)
                self.saved_epochs.append(dataclass.replace(self.epoch))
                context.save_model(model_info, self.last_saved_epoch)

        return SaveCallback(self)
