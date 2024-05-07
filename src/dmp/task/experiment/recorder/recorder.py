from abc import ABC
from typing import Any, Dict, List, Sequence
import tensorflow.keras as keras

from dmp.task.experiment.training_experiment.epoch_counter import EpochCounter
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


class Recorder(keras.callbacks.Callback, ABC):
    def __init__(
        self,
        epoch_counter: EpochCounter,
    ):
        super().__init__()
        self._epoch_counter: EpochCounter = epoch_counter
        self.epoch: List[TrainingEpoch] = []
        self.history: Dict[str, List] = {}

    def _record_metric(self, metric: str, value: Any) -> None:
        self.history.setdefault(metric, []).append(value)

    def _record_epoch(self) -> None:
        self.epoch.append(self._epoch_counter.current_epoch)

    # def on_train_begin(self, logs=None):
    #     super().on_train_begin(logs=logs)
    #     self.epoch = []
    #     self.history = {}
