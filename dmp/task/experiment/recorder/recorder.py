from abc import ABC
from typing import Any, Dict, List, Sequence
import tensorflow.keras as keras


class Recorder(keras.callbacks.Callback, ABC):
    def __init__(self):
        super().__init__()
        self.epoch: List[int] = []
        self.history: Dict[str, List] = {}

    def _record_metric(self, metric: str, value: Any) -> None:
        self.history.setdefault(metric, []).append(value)

    def _record_epoch(self, epoch: int) -> None:
        self.epoch.append(epoch)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.epoch = []
        self.history = {}
