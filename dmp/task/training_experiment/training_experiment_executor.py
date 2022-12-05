from typing import Any, Dict
import os
import platform
import subprocess
import tensorflow
import tensorflow.keras as keras
import numpy

from dmp.data.pmlb import pmlb_loader
from dmp.jobqueue_interface import jobqueue_marshal
from dmp.layer.visitor.keras_interface.layer_to_keras import KerasLayer, make_keras_network_from_layer
import dmp.task.aspect_test.keras_utils as keras_utils
from dmp.task.task_util import remap_key_prefixes
from dmp.layer import *
from dmp.task.training_experiment.additional_validation_sets import AdditionalValidationSets
from dmp.task.training_experiment.training_experiment_utils import *
from dmp.task.training_experiment.training_experiment import TrainingExperiment
from dmp.task.training_experiment.network import Network

test_history_key: str = 'test'


class TrainingExperimentExecutor():

    def __init__(
        self,
        task: TrainingExperiment,
        worker,
        *args,
        **kwargs,
    ) -> None:
        self.set_random_seeds(task)

        (
            ml_task,
            input_shape,
            output_shape,
            fit_config,
            test_data,
        ) = self.load_and_prepare_dataset(task)

        network: Network = self.make_network(
            task,
            input_shape,
            output_shape,
            task.size,
            ml_task,
        )
        network.compile_model(task.optimizer)
        callbacks = self.make_callbacks(task, test_data)
        self.add_early_stopping_callback(task, callbacks)
        history = self.fit_model(fit_config, network, callbacks)
        self.result = self.make_result_record(task, worker, network, history)

    def __call__(self) -> Dict[str, Any]:
        return self.result

    def set_random_seeds(self, task: TrainingExperiment) -> None:
        seed: int = task.seed
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

    def load_dataset(self, task: TrainingExperiment) -> Tuple[Any, Any, Any]:
        dataset_series, inputs, outputs = pmlb_loader.load_dataset(
            pmlb_loader.get_datasets(), task.dataset)
        return dataset_series, inputs, outputs

    def load_and_prepare_dataset(
        self,
        task: TrainingExperiment,
    ) -> Tuple[str, Sequence[int], Sequence[int], Dict[str, Any], Any]:
        # load dataset
        dataset_series, inputs, outputs = self.load_dataset(task)
        ml_task = str(dataset_series['Task'])
        input_shape = inputs.shape
        output_shape = outputs.shape

        (
            train_data,
            validation_data,
            test_data,
        ) = tuple((self.make_tensorflow_dataset(
            dataset,
            task.run_config['batch_size'],
        ) for dataset in split_dataset(
            task.test_split_method,
            task.test_split,
            task.validation_split,
            task.label_noise,
            ml_task,
            inputs,
            outputs,
        )))

        prepared_config = deepcopy(task.run_config)
        prepared_config['x'] = train_data

        if validation_data is None:
            prepared_config['validation_data'] = test_data
            test_data = None
        else:
            prepared_config['validation_data'] = validation_data

        # TODO -- resume here
        return (
            ml_task,
            input_shape,
            output_shape,
            prepared_config,
            test_data,
        )

    def make_tensorflow_dataset(
        self,
        # task: TrainingExperiment,
        datasets: Sequence,
        batch_size: int,
    ) -> Any:
        if datasets[0] is None:
            return None

        dataset_options = tensorflow.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = \
            tensorflow.data.experimental.AutoShardPolicy.DATA

        tf_datasets = tuple((tensorflow.data.Dataset.from_tensor_slices(
            dataset).with_options(dataset_options).astype('float32')
                             for dataset in datasets))

        tf_datasets = tensorflow.data.Dataset.from_tensor_slices(tf_datasets)
        tf_datasets = tf_datasets.with_options(dataset_options)
        tf_datasets = tf_datasets.batch(batch_size)
        return tf_datasets

    def make_network(
        self,
        task: TrainingExperiment,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        target_size: int,
        ml_task: str,
    ) -> Network:
        output_activation, loss = get_output_activation_and_loss_for_ml_task(
            output_shape[1], ml_task)

        # Generate neural network architecture

        # TODO: make it so we don't need this hack
        shape = task.shape
        residual_mode = 'none'
        residual_suffix = '_residual'
        if shape.endswith(residual_suffix):
            residual_mode = 'full'
            shape = shape[0:-len(residual_suffix)]

        # TODO: is this the best way to do this? Maybe these could be passed as a dict?
        layer_args = {
            'kernel_regularizer': task.kernel_regularizer,
            'bias_regularizer': task.bias_regularizer,
            'activity_regularizer': task.activity_regularizer,
            'activation': task.activation,
        }
        if task.batch_norm:  # TODO: add this or use dict to config
            layer_args['batch_norm'] = task.batch_norm

        # Build NetworkModule network
        delta, widths, network_structure, num_free_parameters, layer_shapes = \
            find_best_layout_for_budget_and_depth(
                input_shape,
                residual_mode,
                task.input_activation,
                output_activation,
                target_size,
                widths_factory(shape)(output_shape[1], task.depth),
                layer_args,
            )

        # reject non-conformant network sizes
        delta = num_free_parameters - task.size
        relative_error = delta / task.size
        if numpy.abs(relative_error) > .2:
            raise ValueError(
                f'Could not find conformant network error : {relative_error}%, delta : {delta}, size: {task.size}.'
            )

        # make a keras model from the network structure
        keras_model = None
        with worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)

            # Build Keras model
            (
                keras_inputs,
                keras_outputs,
                layer_to_keras_map,
            ) = make_keras_network_from_layer(network_structure, layer_shapes)

            keras_model = keras.Model(
                inputs=keras_inputs,
                outputs=keras_outputs,
            )

            if len(keras_model.inputs) != 1:  # type: ignore
                raise ValueError('Wrong number of keras inputs generated')

        keras_num_free_parameters = keras_utils.count_trainable_parameters_in_keras_model(
            keras_model)
        if keras_num_free_parameters != num_free_parameters:
            raise RuntimeError('Wrong number of trainable parameters')

        return Network(
            network_structure,
            layer_shapes,
            widths,
            num_free_parameters,
            loss,
            output_activation,
            layer_to_keras_map,
            keras_model,  # type: ignore
        )

    def make_callbacks(
        self,
        task: TrainingExperiment,
        test_data,
    ) -> List[keras.callbacks.Callback]:
        callbacks = []
        if test_data is not None:
            callbacks.append(
                AdditionalValidationSets(
                    [(test_history_key, test_data)],
                    batch_size=task.run_config['batch_size'],
                ))
        return callbacks

    def add_early_stopping_callback(
        self,
        task: TrainingExperiment,
        callbacks: List[keras.callbacks.Callback],
    ) -> None:
        if task.early_stopping is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(**task.early_stopping))

    def fit_model(
        self,
        fit_config: Dict[str, Any],
        network: Network,
        callbacks: List[keras.callbacks.Callback],
    ) -> Dict:
        history = network.keras_model.fit(
            callbacks=callbacks,
            **fit_config,
        ).history  # type: ignore

        # Add test set history into history dict.
        for cb in callbacks:
            if isinstance(cb, AdditionalValidationSets):
                history.update(cb.history)

        return history

    def make_result_record(
        self,
        task: TrainingExperiment,
        worker,
        network: Network,
        history: Dict[str, Any],
    ) -> Dict[str, Any]:
        # rename 'val_' keys to 'test_' and un-prefixed history keys to 'train_'
        if test_history_key in history:
            history = remap_key_prefixes(history, [
                ('val_', 'validation_'),
                (test_history_key + '_', 'test_'),
                ('', 'train_'),
            ])  # type: ignore
        else:
            history = remap_key_prefixes(history, [
                ('val_', 'test_'),
                ('', 'train_'),
            ])  # type: ignore

        parameters: Dict[str, Any] = task.parameters
        parameters['widths'] = network.widths
        parameters['num_free_parameters'] = network.num_free_parameters
        parameters['output_activation'] = network.output_activation
        parameters['network_structure'] = jobqueue_marshal.marshal(
            network.network_structure)
        parameters.update(history)

        parameters.update(worker.worker_info)
        parameters['python_version'] = str(platform.python_version())
        parameters['platform'] = str(platform.platform())
        parameters['tensorflow_version'] = str(tensorflow.__version__)
        git_hash = None
        try:
            git_hash = subprocess.check_output(
                ["git", "describe", "--always"],
                cwd=os.path.dirname(__file__)).strip().decode()
        except:
            pass

        parameters['git_hash'] = git_hash
        parameters['hostname'] = str(platform.node())
        parameters['slurm_job_id'] = os.getenv("SLURM_JOB_ID")
        return parameters
