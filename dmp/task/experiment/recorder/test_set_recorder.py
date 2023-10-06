from abc import ABC
import time
from typing import Any, List, Optional
import tensorflow.keras as keras
from dmp.task.experiment.recorder.recorder import Recorder
from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder

from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.task.experiment.training_experiment import training_experiment_keys


class TestSetRecorder(Recorder, ABC):
    def __init__(
        self,
        test_sets: List[TestSetInfo],
        timestamp_recorder: Optional[TimestampRecorder],
    ):
        super().__init__()
        self._test_sets: List[TestSetInfo] = test_sets
        self._timestamp_recorder: Optional[TimestampRecorder] = timestamp_recorder

    def accumulate_metrics(self, epoch: int) -> None:
        self._record_epoch(epoch)
        model: keras.Model = self.model  # type: ignore

        # evaluate on the additional test sets
        print(
            f"TestSetRecorder: evaluating {len(self._test_sets)} additional test sets..."
        )
        for test_set in self._test_sets:
            if test_set.test_data is None:
                continue
            print(f"TestSetRecorder: evaluating {test_set.history_key}...")
            start_time = time.time()
            results = self._evaluate_set(test_set)
            end_time = time.time()
            print(f"TestSetRecorder: done evaluating {test_set.history_key}.")
            if self._timestamp_recorder is not None:
                self._timestamp_recorder.record_time(
                    test_set.history_key, end_time - start_time
                )
            for metric, result in zip(model.metrics_names, results):
                self._accumulate_test_set_metric(test_set, metric, result)

    def _evaluate_set(self, test_set: TestSetInfo) -> Any:
        model: keras.Model = self.model  # type: ignore
        return model.evaluate(
            x=test_set.test_data,
            y=test_set.test_targets,
            sample_weight=test_set.sample_weights,
            verbose=1,  # type: ignore
            use_multiprocessing=True,
            workers=8,
        )

    def _accumulate_test_set_metric(
        self,
        test_set: TestSetInfo,
        metric,
        result,
    ) -> None:
        self._record_metric(
            test_set.history_key + "_" + metric,
            result,
        )
