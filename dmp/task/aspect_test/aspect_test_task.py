import dataclasses

from dataclasses import dataclass
import os
import platform
import subprocess


from .aspect_test_utils import *
from dmp.task.task import ParameterDict, Task, Parameter

# budget -> size
# topology -> shape
#


@dataclass
class AspectTestTask(Task):
    # Parameters

    # log: str = './log'
    dataset: str
    # test_split: int = 0
    input_activation: str
    activation: str

    optimizer: dict
    # learning_rate = optimizer.config.learning_rate
    # optimizer = optimizer.class_name
    # learning_rate: float

    shape: str
    size: int
    depth: int
    # epoch_scale: dict
    # rep: int

    test_split: float  
    test_split_method: str
    run_config: dict
    # run_config.batch_size
    # run_config.epochs
    # run_config.shuffle

    label_noise: float

    kernel_regularizer : Optional[dict] = None
    bias_regularizer : Optional[dict] = None
    activity_regularizer : Optional[dict] = None

    early_stopping: Optional[dict] = None
    save_every_epochs: Optional[int] = None

    def __call__(self) -> Dict[str, any]:
        from .aspect_test_executor import AspectTestExecutor
        return AspectTestExecutor(*dataclasses.astuple(self))(self)

    @property
    def version(self) -> int:
        return 3

    @property
    def parameters(self) -> ParameterDict:
        parameters = super().parameters

        def rename_param(src, dest):
            if src in parameters:
                parameters[dest] = parameters[src]
                del parameters[src]

        rename_param('optimizer.config.learning_rate', 'learning_rate')
        rename_param('optimizer.class_name', 'optimizer')
        rename_param('run_config.batch_size', 'batch_size')
        rename_param('run_config.epochs', 'epochs')

        parameters.pop('run_config.validation_split', None)
        parameters.pop('run_config.verbose', None)

        return parameters
