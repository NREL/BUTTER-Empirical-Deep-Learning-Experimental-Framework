from collections import Iterable
from typing import Any, List, Optional
import tensorflow.keras as keras
from dmp.task.recorder.timestamp_recorder import TimestampRecorder

from dmp.task.training_experiment.test_set_info import TestSetInfo
from dmp.task.recorder.test_set_recorder import TestSetRecorder


class ZeroEpochRecorder(TestSetRecorder):

    def __init__(
        self,
        test_sets: List[TestSetInfo],
        timestamp_recorder: Optional[TimestampRecorder],
    ):
        super().__init__(test_sets, timestamp_recorder)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.accumulate_metrics(-1)
