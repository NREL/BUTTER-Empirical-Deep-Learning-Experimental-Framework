from dataclasses import dataclass
from dmp.task.aspect_test.aspect_test_task import  AspectTestTask
from typing import Optional

@dataclass
class GrowthTestTask(AspectTestTask):
    growth_trigger: Optional[str] = None
    growth_trigger_params: Optional[dict] = None
    growth_method: Optional[str] = None
    growth_method_params: Optional[dict] = None
    growth_scale: Optional[float] = None
    max_size: Optional[int] = None
    val_split: Optional[float] = None
