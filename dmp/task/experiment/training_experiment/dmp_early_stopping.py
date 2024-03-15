from dataclasses import replace
from typing import Optional
import tensorflow.keras as keras
from dmp.task.experiment.training_experiment.epoch_counter import EpochCounter

from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


class DMPEarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(self, dmp_epoch_counter: EpochCounter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmp_epoch_counter: EpochCounter = dmp_epoch_counter
        self.best_training_epoch: Optional[TrainingEpoch] = None

    def on_train_end(self, *args, **kwargs):
        super().on_train_end(*args, **kwargs)

        if self.restore_best_weights and self.best_weights is not None:
            # restore best weights if super() did not
            if self.stopped_epoch <= 0:
                self.model.set_weights(self.best_weights)

            # set dmp_epoch to match best weights epoch
            counter = self.dmp_epoch_counter
            counter.current_epoch = replace(self.dmp_epoch_counter.initial_epoch)
            counter.current_epoch.count_new_epoch(self.best_epoch + 1)
            counter.current_epoch.marker = 1
            self.best_training_epoch = counter.current_epoch
