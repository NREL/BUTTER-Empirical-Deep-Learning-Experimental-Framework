from collections import Iterable
from typing import Any, List
import tensorflow.keras as keras

from dmp.task.training_experiment.test_set_info import TestSetInfo
from dmp.task.training_experiment.test_set_recorder import TestSetRecorder


class ZeroEpochRecorder(TestSetRecorder):

    def __init__(self, test_sets: List[TestSetInfo]):
        super().__init__(test_sets)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.accumulate_metrics(0)
