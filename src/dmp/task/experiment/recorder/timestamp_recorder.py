from abc import ABC
from typing import Any, List
import tensorflow.keras as keras
import time
from dmp.task.experiment.recorder.recorder import Recorder
from dmp.task.experiment.training_experiment.epoch_counter import EpochCounter


class TimestampRecorder(Recorder):
    def __init__(
        self,
        epoch_counter: EpochCounter,
        time_suffix: str,
        epoch_start_metric_name: str,
        epoch_duration_metric_name: str,
    ):
        super().__init__(epoch_counter)
        self._epoch_start_time: float = -1.0
        self._time_suffix: str = time_suffix
        self._epoch_start_metric_name: str = epoch_start_metric_name
        self._epoch_duration_metric_name: str = epoch_duration_metric_name

    def record_time(
        self,
        metric_name: str,
        seconds: float,
    ) -> None:
        relative_ms = int(seconds * 1000)
        self._record_metric(metric_name + self._time_suffix, relative_ms)

    def on_epoch_begin(self, epoch, logs=None):
        self._record_epoch()
        self._epoch_start_time = time.time()
        self.record_time(
            self._epoch_start_metric_name,
            self._epoch_start_time,  # - 1689000000,
        )

    def on_epoch_end(self, epoch, logs=None):
        epoch_end = time.time()
        self.record_time(
            self._epoch_duration_metric_name,
            epoch_end - self._epoch_start_time,
        )
