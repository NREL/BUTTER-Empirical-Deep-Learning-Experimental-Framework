from abc import ABC
from typing import Any, List
import tensorflow.keras as keras
import time
from dmp.task.experiment.recorder.recorder import Recorder

from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.task.experiment.training_experiment import training_experiment_keys


class TimestampRecorder(Recorder):

    def __init__(self):
        super().__init__()
        self.epoch_start_time: float = -1.0

    def record_interval(
        self,
        metric_name: str,
        seconds: float,
    ) -> None:
        relative_ms = round(seconds * 1000)
        self._record_metric(metric_name, relative_ms)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._record_epoch(epoch)
        self.epoch_start_time = time.time()
        self.record_interval(
            training_experiment_keys.keys.epoch_start_time_ms,
            self.epoch_start_time,
        )

    def on_epoch_end(self, epoch, logs=None):
        epoch_end = time.time()
        self.record_interval(
            training_experiment_keys.keys.epoch_time_ms,
            epoch_end - self.epoch_start_time,
        )
