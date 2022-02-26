import dataclasses
from dataclasses import field
from os import environ
import time
from typing import Type

from dataclasses import dataclass


from .aspect_test_utils import *
from dmp.task.task import Task
from dmp.record.base_record import BaseRecord
from dmp.record.history_record import HistoryRecord
from dmp.record.val_loss_record import ValLossRecord

import pandas
import numpy
import uuid


@dataclass
class AspectTestTask(Task):
    # Parameters

    # log: str = './log'
    dataset: str
    # test_split: int = 0
    input_activation: str
    activation: str

    optimizer: dict
    # learning_rate = optimizer.config.learning_rate
    # optimizer = optimizer.class_name
    # learning_rate: float

    topology: str
    residual_mode: str
    budget: int
    depth: int
    # epoch_scale: dict
    # rep: int

    validation_split: float  # does not use run_config.validation_split
    validation_split_method: str
    run_config: dict
    # run_config.batch_size
    # run_config.epochs
    # run_config.shuffle

    label_noise: float

    early_stopping: Optional[dict] = None
    save_every_epochs: Optional[int] = None
    
    def __call__(self) -> None:
        from .aspect_test_executor import AspectTestExecutor
        return AspectTestExecutor(
            **dataclasses.asdict(self)
        )()

