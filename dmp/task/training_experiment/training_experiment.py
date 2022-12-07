from dataclasses import dataclass
from typing import Any, Optional, Dict
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec
from dmp.task.task import Parameter, ParameterDict, Task


@dataclass
class TrainingExperiment(Task):
    dataset: DatasetSpec  # migrate dataset stuff into here
    model: ModelSpec  # defines network
    fit_config: dict  # contains batch size, epochs, shuffle (migrate from run_config)
    optimizer: dict  # contains learning rate (migrate converting to typed config from keras serialization)
    loss: dict  # migrate from runtime (converting from simple string to typed config)
    early_stopping: Optional[dict]  # direct migration
    save_every_epochs: int  # migrate with None mapping to -1

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

        # rename_param('optimizer.config.learning_rate', 'learning_rate') # migrate to optimizer.learning_rate
        # rename_param('optimizer.class_name', 'optimizer') # migrate to optimizer.type
        # rename_param('run_config.batch_size', 'batch_size') # migrate to fit_config.batch_size
        # rename_param('run_config.epochs', 'epochs') # migrate to fit_config.epochs

        # parameters.pop('run_config.validation_split', None) #??
        # parameters.pop('run_config.verbose', None) #??

        return parameters


'''
    + what if attributes are more free-form?
        + could define experiment with a more minimal set of parameters
            + and compute other attributes at runtime (and/or some simple pass-through)
            
    + optimizer : optimizer config (same) # contains learning rate
    + dataset : str (same)
    + test_split : float (same)
    + split_method : str (renamed from test_split_method)
    + run_config : dict (same) # contains batch size, epochs, shuffle
    + label_noise : (same)
    + early_stopping : (same)
    + save_every_epochs : (same) (TODO: make sure not part of experiment table)

    + network : NetworkSpecification -> defines and creates network
        + input_shape : Sequence[int] (migrate from runtime)
        + output_activation ??
        + loss (migrate from runtime calculation)
            output_activation, loss = get_output_activation_and_loss_for_ml_task(
            dataset.output_shape[1], dataset.ml_task)
        + DenseBySizeAndShape
            + shape (migrate)
            + size (migrate)
            + depth (migrate)
            + layer : dict
                + activation (migrate)
                + kernel_regularizer (migrate)
                + bias_regularizer (migrate)
                + activity_regularizer (migrate)
            + output_layer : dict 
                + activation (migrate from runtime compute of output_activation)
                + units/shape (migrate from runtime)
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