from dataclasses import dataclass

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


@dataclass
class PruningConfig:
    iterations: int  # how many times to prune
    method: PruningMethod  # how to prune
    max_epochs_per_iteration: int  # max training epochs between prunings if early stopping isn't triggered
    rewind_epoch: TrainingEpoch  # epoch to rewind to after pruning
    rewind_optimizer: bool  # rewind optimizer state too?
    new_seed: bool  # set to True iff the parent run used a different seed
 