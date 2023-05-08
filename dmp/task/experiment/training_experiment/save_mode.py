from abc import ABC, abstractmethod

from typing import Any, Optional

from jobqueue.job import Job
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.worker import Worker
from dmp.model.model_info import ModelInfo


class SaveMode(ABC):
    '''
    Configures the model saving policy during an experiment.
    '''

    @abstractmethod
    def make_save_model_callback(
        self,
        worker: Worker,
        job: Job,
        task: ExperimentTask,
    ):
        pass
