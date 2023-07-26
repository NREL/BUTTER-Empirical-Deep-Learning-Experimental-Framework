from dataclasses import dataclass
from uuid import UUID

from dmp.task.experiment.training_experiment.run_spec import RunSpec


@dataclass
class IterativePruningRunSpec(RunSpec):
    rewind_run_id: UUID  # run to rewind weights from
    prune_first_iteration: bool  # if True, the model will be trained before the first pruning, if False, the model will be pruned before any training
