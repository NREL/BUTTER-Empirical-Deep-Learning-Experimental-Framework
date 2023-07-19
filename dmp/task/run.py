from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from dmp.task.experiment.experiment import Experiment
from dmp.task.experiment.training_experiment.run_spec import RunSpec

from typing import TYPE_CHECKING

from dmp.task.task import Task

if TYPE_CHECKING:
    from dmp.context import Context


@dataclass
class Run(Task):
    experiment: Experiment
    run: RunSpec

    def __call__(
        self,
        context: Context,
    ) -> None:
        self.experiment(context, self.run)
