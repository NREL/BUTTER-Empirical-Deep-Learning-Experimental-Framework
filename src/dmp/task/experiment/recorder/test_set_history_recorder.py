from typing import Any, List, Optional
import tensorflow.keras as keras
from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder
from dmp.task.experiment.training_experiment.epoch_counter import EpochCounter

from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder


class TestSetHistoryRecorder(TestSetRecorder):
    def __init__(
        self,
        epoch_counter: EpochCounter,
        test_sets: List[TestSetInfo],
        timestamp_recorder: Optional[TimestampRecorder],
    ):
        super().__init__(epoch_counter, test_sets, timestamp_recorder)

    def on_epoch_end(self, epoch, logs=None):
        self.accumulate_metrics()
