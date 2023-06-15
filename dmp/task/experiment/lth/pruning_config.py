

from dataclasses import dataclass
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import PruningMethod


@dataclass
class PruningConfig():
    pruning_iteration: int
    max_iterations: int # run-specific
    method: PruningMethod
