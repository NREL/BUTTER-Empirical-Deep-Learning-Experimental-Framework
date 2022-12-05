from dataclasses import dataclass
from typing import Any, Optional, Dict

from dmp.task.task import Parameter, ParameterDict, Task


@dataclass
class TrainingExperiment(Task):
    dataset: str
    test_split_method: str
    test_split: float
    validation_split: float
    label_noise: float

    run_config: dict  # contains batch size, epochs, shuffle
    optimizer: dict  # contains learning rate
    early_stopping: Optional[dict]
    save_every_epochs: int

    network : dict # defines network

    def __call__(self, worker, *args, **kwargs) -> Dict[str, Any]:
        from .training_experiment_executor import TrainingExperimentExecutor
        return TrainingExperimentExecutor(self, worker, *args, **kwargs)()

    @property
    def version(self) -> int:
        return 0

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


'''
    + what if attributes are more free-form?
        + could define experiment with a more minimal set of parameters
            + and compute other attributes at runtime (and/or some simple pass-through)
            
    + optimizer : optimizer config (same) # contains learning rate
    + dataset : str (same)
    + test_split : float (same)
    + test_split_method : str (same)
    + run_config : dict (same) # contains batch size, epochs, shuffle
    + label_noise : (same)
    + early_stopping : (same)
    + save_every_epochs : (same) (TODO: make sure not part of experiment table)

    + network : NetSpec -> defines and creates network
        + DenseBySizeAndShape
            + shape (migrate)
            + size (migrate)
            + depth (migrate)
            + input_layer : dict
                + activation (migrate from input_activation)
            + layer : dict
                + activation (migrate)
                + kernel_regularizer (migrate)
                + bias_regularizer (migrate)
                + activity_regularizer (migrate)
            + output_layer : dict 
                + activation (migrate from runtime compute of output_activation)
        + CNNStackAndDownsample
            + num_stacks
            + cells_per_stack
            + stem : dict
            + cell_operations: List[List[str]] (and/or preset operations name?)
            + cell_conv : dict
            + cell_pooling : dict
            + downsample_conv : dict
            + downsample_pooling : dict
            + output : dict
'''