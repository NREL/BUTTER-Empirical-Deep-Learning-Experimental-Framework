

from dataclasses import dataclass
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import PruningMethod


@dataclass
class PruningConfig():
    num_pruning_iterations: int
    pruning_iteration: int
    method: PruningMethod
