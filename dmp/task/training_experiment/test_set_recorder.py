from abc import ABC
from typing import Any, List
import tensorflow.keras as keras

from dmp.task.training_experiment.test_set_info import TestSetInfo


class TestSetRecorder(keras.callbacks.Callback, ABC):

    def __init__(self, test_sets:List[TestSetInfo]):
        """
        Adapted From : https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
        """
        super().__init__()

        self.test_sets: List[TestSetInfo] = test_sets
        # self.epoch_counter : int = 0
        self.epoch: List[int] = []
        self.history: dict = {}

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.epoch = []
        self.history = {}

    # def on_train_begin(self, logs=None):
    #     pass

    def accumulate_metrics(self, epoch: int) -> None:
        self.epoch.append(epoch)
        model: keras.Model = self.model  # type: ignore

        # evaluate on the additional test sets
        for test_set in self.test_sets:
            results = self.evaluate_set(test_set)
            for metric, result in zip(model.metrics_names, results):
                self.accumulate_metric(test_set, metric, result)

    def evaluate_set(self, test_set: TestSetInfo) -> Any:
        model: keras.Model = self.model  # type: ignore
        return model.evaluate(
            x=test_set.test_data,
            y=test_set.test_targets,
            sample_weight=test_set.sample_weights,
        )

    def accumulate_metric(
        self,
        test_set: TestSetInfo,
        metric,
        result,
    ) -> None:
        valuename = test_set.history_key + '_' + metric
        self.history.setdefault(valuename, []).append(result)
