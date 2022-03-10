import dataclasses

from dataclasses import dataclass


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

    validation_split: float  # does not use run_config.validation_split
    validation_split_method: str
    run_config: dict
    # run_config.batch_size
    # run_config.epochs
    # run_config.shuffle

    label_noise: float

    early_stopping: Optional[dict] = None
    save_every_epochs: Optional[int] = None
    
    def __call__(self) -> Tuple[Dict[str, Parameter], Dict[str, any]]:
        from .aspect_test_executor import AspectTestExecutor
        return AspectTestExecutor(
            **dataclasses.asdict(self)
        )()

    @property
    def _run_value_keys(self) -> List[str]:
        return super()._run_value_keys + ['save_every_epochs']

    @property
    def parameters(self) -> ParameterDict:
        experiment_parameters, run_parameters, run_values = super().parameters

        def rename_param(src, dest):
            if src in experiment_parameters:
                experiment_parameters[dest] = experiment_parameters[src]
                del experiment_parameters[src]

        rename_param('optimizer.config.learning_rate', 'learning_rate')
        rename_param('optimizer.class_name', 'optimizer')
        rename_param('run_config.batch_size', 'batch_size')
        rename_param('run_config.epochs', 'epochs')

        experiment_parameters.pop('run_config.validation_split', None)
        experiment_parameters.pop('run_config.verbose', None)

        return experiment_parameters, run_parameters, run_values

        



    

