import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from dmp.task.task import register_task_type
from dmp.task.task_result_record import TaskResultRecord
from dmp.task.training_experiment.training_experiment import TrainingExperiment


@dataclass
class GrowthExperiment(TrainingExperiment):

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
    # num_scales: int = 1024
    initial_size: int = 1024
    max_total_epochs: int = 3000
    max_equivalent_epoch_budget: int = 3000

    def __call__(self, worker, *args, **kwargs) -> TaskResultRecord:
        from .growth_experiment_executor import GrowthExperimentExecutor
        return GrowthExperimentExecutor(self, worker, *args, **kwargs)()


register_task_type(GrowthExperiment)