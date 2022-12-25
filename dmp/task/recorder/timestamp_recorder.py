from abc import ABC
from typing import Any, List
import tensorflow.keras as keras
import time
from dmp.task.recorder.recorder import Recorder

from dmp.task.training_experiment.test_set_info import TestSetInfo
from dmp.task.training_experiment.training_experiment_keys import TrainingExperimentKeys


class TimestampRecorder(Recorder):

    def __init__(self):
        super().__init__()
        # self.train_start_time: float = -1.0
        self.epoch_start_time: float = -1.0

    # def _delta_time(self) -> float:
    #     return time.time() - self.train_start_time

    # def record_delta_time(self, metric_name: str) -> float:
    #     delta_time = self._delta_time()
    #     self.history.setdefault(metric_name, []).append(delta_time)
    #     return delta_time

    def record_interval(self, metric_name:str, seconds:float,)->None:
        relative_ms = round(seconds * 1000)
        self._record_metric(metric_name, relative_ms)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        # self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._record_epoch(epoch)
        self.epoch_start_time = time.time()
        self.record_interval(
            TrainingExperimentKeys.epoch_start_time_ms,
            self.epoch_start_time)
            # self.epoch_start_time - self.train_start_time)

    def on_epoch_end(self, epoch, logs=None):
        epoch_end = time.time()
        self.record_interval(
            TrainingExperimentKeys.epoch_time_ms,
            epoch_end - self.epoch_start_time)
