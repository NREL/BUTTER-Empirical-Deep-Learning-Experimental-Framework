import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from dmp.task.training_experiment.training_experiment import TrainingExperiment


@dataclass
class GrowthExperiment(TrainingExperiment):

    # val_split: float = .1 # moved to TrainingExperiment
    # growth_trigger: str = 'EarlyStopping'
    growth_trigger: dict = \
        field(default_factory=lambda: {
            'type': 'EarlyStopping',
            'restore_best_weights' : True,
            })
    growth_method: dict = \
        field(default_factory=lambda: {
            'type': 'NetworkOverlayer',
            })
    growth_scale: float = 2.0
    # initial_size: int = 1024
    max_total_epochs: int = 3000
    max_equivalent_epoch_budget: int = 3000

    def __call__(self, worker, *args, **kwargs) -> Dict[str, Any]:
        from .growth_experiment_executor import GrowthExperimentExecutor
        return GrowthExperimentExecutor(self, worker, *args, **kwargs)()
