from dataclasses import replace
import tensorflow.keras as keras

from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


class EpochCounter(keras.callbacks.Callback):
    def __init__(self, initial_epoch: TrainingEpoch):
        super().__init__()
        self.initial_epoch: TrainingEpoch = replace(initial_epoch)
        self.training_epoch: TrainingEpoch = replace(initial_epoch)
        self.new_fit_number: bool = True

    def on_train_begin(self, *args, **kwargs) -> None:
        if self.new_fit_number:
            self.initial_epoch = replace(self.training_epoch)
            self.initial_epoch.count_new_model()
            self.training_epoch = replace(self.initial_epoch)

    def on_epoch_end(self, *args, **kwargs) -> None:
        self.training_epoch = replace(self.training_epoch)
        self.training_epoch.count_new_epoch()