from dataclasses import dataclass
from typing import Any

from dmp.task.task import Parameter, ParameterDict, Task

from ..aspect_test.aspect_test_utils import *

# budget -> size
# topology -> shape
#


@dataclass
class SizeBasedExperiment(Task):
    dataset: str

    input_activation: str

    network_config: dict
    '''
        + what if attributes are more free-form?
            + could define experiment with a more minimal set of parameters
                + and compute other attributes at runtime (and/or some simple pass-through)
            
        + network_graph : Layer graph of network? (migrate from runtime, NetworkModule graph)
        
        + optimizer : optimizer config (same) # contains learning rate
        + dataset : str (same)
        + test_split : float (same)
        + test_split_method : str (same)
        + run_config : dict (same) # contains batch size, epochs, shuffle
        + label_noise : (same)
        + early_stopping : (same)
        + save_every_epochs : (same) (TODO: make sure not part of experiment table)

        + network : NetSpec -> defines and creates network
            + DenseFullyConnected
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

        
        + cnn config: where does this info go?

    '''

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
        from ..aspect_test.aspect_test_executor import AspectTestExecutor
        return AspectTestExecutor()(self, worker, *args, **kwargs)

    @property
    def version(self) -> int:
        return 4

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
