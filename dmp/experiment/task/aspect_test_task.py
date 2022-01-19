import dataclasses
from dataclasses import field
from os import environ
import time
from typing import Type

from attr import dataclass

from dmp.experiment.task.aspect_test import AspectTest

from .aspect_test_utils import *
from dmp.experiment.batch.batch import CartesianBatch
from dmp.experiment.task.task import Task
from dmp.record.base_record import BaseRecord
from dmp.record.history_record import HistoryRecord
from dmp.record.val_loss_record import ValLossRecord

import pandas
import numpy
import uuid


@dataclass
class AspectTestTask(Task):
    # Parameters
    seed: Optional[int]
    # log: str = './log'
    dataset: str
    # test_split: int = 0
    input_activation: str
    internal_activation: str
    optimizer: dict
    # learning_rate: float = None
    topology: str
    residual_mode: str
    budget: int
    depth: int
    # epoch_scale: dict
    # rep: int
    early_stopping: Optional[dict]
    validation_split: float
    run_config: dict
    checkpoint_epochs: Optional[int]
    validation_split_method: Optional[str]
    label_noise: Optional[Union[float, str]]

    def __call__(self, job_id: uuid.UUID) -> None:
        AspectTest(self, job_id)
