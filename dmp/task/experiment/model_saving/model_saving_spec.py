from dataclasses import dataclass, replace
import math
from typing import Any, Dict, List, Optional, Sequence, Set

from dmp.model.model_info import ModelInfo
from dmp.task.experiment.training_experiment.epoch_counter import EpochCounter
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.context import Context
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class ModelSavingSpec:
    """
    Saves every fixed_interval steps, up to the fixed_threshold step, and then saves every exponential_rate ^ i steps, where i is a positive integer and exponential_rate ^ i >= fixed_threshold.
    To only save initial and/or final models, set fixed_threshold = 0, and exponential_rate = 0.0.
    To disable fixed savepoint spacing set fixed_threshold = 0.
    To use only fixed spacing, set fixed_threshold = -1.
    To disable exponential savepoint spacing set exponential_rate = 0.0.
    """

    save_initial_model: bool
    save_trained_model: bool

    save_epochs: List[int]  # specific global epochs to save at
    save_fit_epochs: List[int]

    fixed_interval: int
    fixed_threshold: int
    exponential_rate: float

    # specific model epochs to save at for every model

    def make_save_model_callback(
        self,
        context: Context,
        epoch_counter: EpochCounter,
        model_info: ModelInfo,
        # checkpoint_interval: float,
    ):
        import time
        from dmp.task.experiment.model_saving.model_saving_callback import (
            ModelSavingCallback,
        )

        class SaveCallback(ModelSavingCallback):
            def __init__(self, parent: ModelSavingSpec):
                super().__init__()
                self.parent: ModelSavingSpec = parent
                self.save_fit_epochs: Set[int] = set(parent.save_fit_epochs)
                self.save_epochs: Set[int] = set(parent.save_epochs)

                # self.epoch: TrainingEpoch = replace(epoch)
                # self.epoch.fit_number -= 1

                self.last_saved_epoch: TrainingEpoch = replace(
                    epoch_counter.training_epoch
                )
                self.last_saved_epoch.epoch -= 1
                self.last_saved_epoch.fit_epoch -= 1

                self.last_save_time: float = time.time()

                # self.history : Dict[str, List] = {}
                self.checkpoints: List[TrainingExperimentCheckpoint] = []

            def saved_epochs(self) -> Sequence[TrainingEpoch]:
                return [checkpoint.epoch for checkpoint in self.checkpoints]

            def on_train_begin(self, logs=None) -> None:
                if self.parent.save_initial_model:
                    self.save_model()

            def on_epoch_end(self, epoch, logs=None) -> None:
                training_epoch = epoch_counter.training_epoch
                global_epoch = training_epoch.epoch
                fit_epoch = training_epoch.fit_epoch

                parent = self.parent
                if (
                    global_epoch in self.save_epochs
                    or fit_epoch in self.save_fit_epochs
                ):
                    # specified epoch
                    pass
                elif parent.fixed_threshold == -1 or (
                    parent.fixed_threshold > 0 and fit_epoch <= parent.fixed_threshold
                ):
                    # fixed regime
                    if fit_epoch % parent.fixed_interval != 0:
                        return
                elif parent.exponential_rate <= 0.0:
                    # exponential regime disabled
                    return
                else:
                    # exponential regime
                    denom = math.log(self.parent.exponential_rate)
                    ratio = math.ceil(math.log(fit_epoch) / denom)
                    next_ratio = math.ceil(math.log(fit_epoch + 1) / denom)
                    if ratio == next_ratio:
                        return

                self.save_model()

            def on_train_end(self, logs=None) -> None:
                if self.parent.save_trained_model:
                    self.save_model(train_end=True)

            def save_model(self, train_end: bool = False) -> None:
                training_epoch = epoch_counter.training_epoch
                if self.last_saved_epoch == training_epoch:
                    return

                self.last_saved_epoch = replace(training_epoch)
                checkpoint = context.save_model(model_info, training_epoch)
                self.checkpoints.append(checkpoint)

        return SaveCallback(self)
