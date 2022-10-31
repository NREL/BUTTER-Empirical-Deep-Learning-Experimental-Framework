import dataclasses
from dataclasses import dataclass
from dmp.task.aspect_test.aspect_test_task import  AspectTestTask
from typing import Optional,Any,Dict

from .growth_test_utils import *

@dataclass
class GrowthTestTask(AspectTestTask):
    asdf: Optional[int] = 1
    val_split:  Optional[float] = None
    growth_trigger: Optional[str] = None
    growth_trigger_params: Optional[dict] = None
    growth_method: Optional[str] = None
    growth_method_params: Optional[dict] = None
    growth_scale: Optional[float] = None
    max_size: Optional[int] = None


    def __call__(self, worker, *args, **kwargs) -> Dict[str, Any]:
        from .growth_test_executor import GrowthTestExecutor
        return GrowthTestExecutor(*dataclasses.astuple(self))(self, worker, *args, **kwargs)
