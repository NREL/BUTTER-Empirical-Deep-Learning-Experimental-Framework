import tensorflow.keras as keras
import numpy
from tensorflow.python.platform import tf_logging as logging


class ProportionalStopping(keras.callbacks.EarlyStopping):
    """ """

    def __init__(
        self,
        min_delta=0.1,
        mode="min",
        **kwargs,
    ):
        super().__init__(mode=mode, **kwargs)
        self.best = None

        if mode == "min":
            self.min_delta = 1.0 - min_delta
        elif mode == "max":
            self.min_delta = 1.0 + min_delta
        else:
            raise ValueError(f"Invalid mode {mode}.")

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None

    def _is_improvement(self, monitor_value, reference_value):
        if reference_value is None:
            return True
        return self.monitor_op(
            monitor_value,
            reference_value * self.min_delta,
        )
        # if not self.monitor_op(monitor_value, reference_value):
        #     return False
        # return mi
        # assert(False)
        # logging.warn(
        #     f'{self.min_delta} {monitor_value} {reference_value} {monitor_value/reference_value}'
        # )
        # return self.monitor_op(self.min_delta, monitor_value / reference_value)
        # return self.monitor_op(monitor_value - self.min_delta, reference_value)
