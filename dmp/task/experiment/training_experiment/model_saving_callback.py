from abc import ABC, abstractmethod
from typing import Sequence
import tensorflow.keras as keras

from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


class ModelSavingCallback(keras.callbacks.Callback, ABC):
    @property
    @abstractmethod
    def saved_epochs(self) -> Sequence[TrainingEpoch]:
        pass
