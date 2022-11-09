import dataclasses
from dataclasses import dataclass, field
from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from typing import Optional, Any, Dict

from .growth_test_utils import *


@dataclass
class GrowthTestTask(AspectTestTask):
    val_split:  float = .1
    growth_trigger: str = 'EarlyStopping'
    growth_trigger_params: dict = field(default_factory=dict)
    growth_method: str = 'grow_network'
    growth_method_params: dict = field(default_factory=dict)
    growth_scale: float = 2.0
    initial_size: int = 1024
    max_total_epochs: int = 3000
    max_equivalent_epoch_budget: int = 3000

    def __call__(self, worker, *args, **kwargs) -> Dict[str, Any]:
        from .growth_test_executor import GrowthTestExecutor
        return GrowthTestExecutor(*dataclasses.astuple(self))(self, worker, *args, **kwargs)
