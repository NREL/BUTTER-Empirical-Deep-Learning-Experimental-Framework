from typing import Any, List
import tensorflow.keras as keras

from dmp.task.training_experiment.test_set_info import TestSetInfo
from dmp.task.training_experiment.test_set_recorder import TestSetRecorder


class TestSetHistoryRecorder(TestSetRecorder):

    def __init__(self, test_sets: List[TestSetInfo]):
        super().__init__(test_sets)

    def on_epoch_end(self, epoch, logs=None):
        self.accumulate_metrics(epoch+1)
