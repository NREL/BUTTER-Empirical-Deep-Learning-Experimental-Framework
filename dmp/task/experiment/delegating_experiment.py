

from abc import ABC
from dataclasses import dataclass
from typing import Optional
from dmp.task.experiment.a_experiment_task import AExperimentTask, FlatParameterDict


@dataclass
class DelegatingExperiment(AExperimentTask, ABC):
    
    delegate : AExperimentTask

    @property
    def version(self) -> int:
        return super().version + 1
    
    @property
    def batch(self)-> str:
        return self.delegate.batch

    @property
    def tags(self)-> Optional[FlatParameterDict]:
        return self.delegate.tags

    @property
    def run_tags(self)-> Optional[FlatParameterDict]:
        return self.delegate.run_tags