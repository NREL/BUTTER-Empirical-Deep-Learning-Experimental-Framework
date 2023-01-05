from dataclasses import dataclass
from typing import Any, Dict, Optional

from dmp.task.task import ParameterDict, Task, register_task_type

@dataclass
class AspectTestTask(Task):
    dataset: str
    input_activation: str
    activation: str

    optimizer: dict  # contains learning rate

    shape: str
    size: int
    depth: int

    test_split: float
    test_split_method: str
    run_config: dict  # contains batch size, epochs, shuffle

    label_noise: float

    kernel_regularizer: Optional[dict] = None
    bias_regularizer: Optional[dict] = None
    activity_regularizer: Optional[dict] = None

    early_stopping: Optional[dict] = None
    save_every_epochs: Optional[int] = None

    def __call__(self, worker, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def version(self) -> int:
        return 4

    # @property
    # def parameters(self) -> ParameterDict:
    #     parameters = super().parameters

    #     def rename_param(src, dst):
    #         if src in parameters:
    #             parameters[dst] = parameters[src]
    #             del parameters[src]

    #     rename_param('optimizer.config.learning_rate', 'learning_rate')
    #     rename_param('optimizer.class_name', 'optimizer')
    #     rename_param('run_config.batch_size', 'batch_size')
    #     rename_param('run_config.epochs', 'epochs')

    #     parameters.pop('run_config.validation_split', None)
    #     parameters.pop('run_config.verbose', None)

    #     return parameters

register_task_type(AspectTestTask)