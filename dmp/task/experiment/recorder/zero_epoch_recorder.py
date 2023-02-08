from typing import Any, List, Optional, Iterable
from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder

from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder


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
