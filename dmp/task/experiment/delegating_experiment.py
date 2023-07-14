from abc import ABC, abstractmethod
from typing import Optional
from dmp.task.experiment.a_experiment_task import AExperimentTask, FlatParameterDict


class DelegatingExperiment(AExperimentTask, ABC):
    @property
    @abstractmethod
    def delegate(self) -> AExperimentTask:
        pass

    @property
    def version(self) -> int:
        return super().version + 1

    @property
    def batch(self) -> str:
        return self.delegate.batch

    @property
    def tags(self) -> Optional[FlatParameterDict]:
        return self.delegate.tags

    @property
    def run_tags(self) -> Optional[FlatParameterDict]:
        return self.delegate.run_tags
