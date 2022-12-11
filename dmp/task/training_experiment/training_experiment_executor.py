from copy import deepcopy
import random
from typing import Any, Dict
import os
import platform
import subprocess
from prometheus_client import Metric
import tensorflow
import tensorflow.keras as keras
import numpy

from dmp.jobqueue_interface import jobqueue_marshal
from dmp.layer.visitor.keras_interface.keras_utils import keras_from_config, make_keras_config
from dmp.layer.visitor.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.dataset.dataset import Dataset
from dmp.layer import *
from dmp.model.model_spec import ModelSpec
from dmp.task.training_experiment.additional_validation_sets import AdditionalValidationSets
from dmp.task.task_util import *
from dmp.task.training_experiment.training_experiment import TrainingExperiment
from dmp.model.network_info import NetworkInfo
from dmp.model.model_info import ModelInfo

test_history_key: str = 'test'


class TrainingExperimentExecutor():

    def __init__(self, task: TrainingExperiment, worker) -> None:
        self.task: TrainingExperiment = task
        self.worker = worker

    def __call__(self) -> Dict[str, Any]:
        self._set_random_seeds()
        dataset = self._load_and_prepare_dataset()
        metrics = self._autoconfigure_for_dataset(dataset)
        model = self._make_model(self.task.model)
        self._compile_model(dataset, model, metrics)
        history = self._fit_model(
            self.task.fit_config,
            dataset,
            model,
            self._make_callbacks(),
        )
        return self._make_result_record(dataset, model, history)

    def _set_random_seeds(self) -> None:
        seed: int = self.task.seed
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

    def _load_and_prepare_dataset(self) -> Dataset:
        return Dataset.make(
            self.task.dataset,
            self.task.fit_config['batch_size'],
        )

    def _autoconfigure_for_dataset(
        self,
        dataset: Dataset,
    ) -> List[Union[str, keras.metrics.Metric]]:
        # auto-populate model inputs and outputs if not already set
        num_outputs: int = dataset.output_shape[0]
        ml_task: str = dataset.ml_task

        metrics = [
            'accuracy',
            keras.metrics.CosineSimilarity(),
            keras.metrics.KLDivergence(),
        ]
        output_kernel_initializer = 'HeUniform'
        output_activation = 'relu'
        if ml_task == 'regression':
            output_activation = 'sigmoid'
            output_kernel_initializer = 'GlorotUniform'
            loss = 'MeanSquaredError'
            metrics.extend([
                keras.metrics.MeanSquaredError(),
                keras.metrics.RootMeanSquaredError(),
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredLogarithmicError(),
            ])
        elif ml_task == 'classification':
            if num_outputs == 1:
                output_activation = 'sigmoid'
                output_kernel_initializer = 'GlorotUniform'
                loss = 'BinaryCrossentropy'
                metrics.extend([
                    keras.metrics.BinaryCrossentropy(),
                    'accuracy',
                    keras.metrics.Hinge(),
                    keras.metrics.SquaredHinge(),
                ])
            else:
                output_activation = 'softmax'
                output_kernel_initializer = 'GlorotUniform'
                loss = 'CategoricalCrossentropy'
                metrics.extend([
                    keras.metrics.CategoricalCrossentropy(),
                    'accuracy',
                    keras.metrics.CategoricalHinge(),
                ])
        else:
            raise Exception('Unknown task "{}"'.format(ml_task))

        model = self.task.model
        if model.input is None:
            model.input = Input()
        if model.input.get('shape', None) is None:
            model.input['shape'] = dataset.input_shape

        if model.output is None:
            model.output = Dense.make(
                dataset.output_shape[0],
                {
                    'activation': None,
                    'kernel_initializer': None,
                },
            )

        output = model.output
        if isinstance(output, Dense):
            if output.get('units', None) is None:
                output['units'] = dataset.output_shape[0]
            if output.get('activation', None) is None:
                output['activation'] = make_keras_config(output_activation)
            if output.get('kernel_initializer', None) is None:
                output['kernel_initializer'] = make_keras_config(
                    output_kernel_initializer)

        if self.task.loss is None:
            self.task.loss = make_keras_config(loss)

        return metrics

    def _make_model(self, model_spec: ModelSpec) -> ModelInfo:
        return self._make_model_from_network(self._make_network(model_spec))

    def _make_network(self, model_spec: ModelSpec) -> NetworkInfo:
        return model_spec.make_network()

    def _make_model_from_network(self, network: NetworkInfo):
        with worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)
            return make_keras_model_from_network(network)

    def _compile_model(
        self,
        dataset: Dataset,
        model: ModelInfo,
        metrics: List[Union[str, keras.metrics.Metric]],
    ) -> None:
        model.keras_model.compile(
            loss=keras_from_config(self.task.loss),  # type: ignore
            optimizer=keras_from_config(self.task.optimizer),
            metrics=metrics,
            run_eagerly=False,
        )

    def _make_callbacks(self) -> List[keras.callbacks.Callback]:
        if self.task.early_stopping is None:
            return []
        return [keras.callbacks.EarlyStopping(**self.task.early_stopping)]

    def _fit_model(
        self,
        fit_config: Dict[str, Any],
        dataset: Dataset,
        model: ModelInfo,
        callbacks: List[keras.callbacks.Callback],
    ) -> Dict:
        # setup training, validation, and test datasets
        fit_config = fit_config.copy()
        fit_config['x'] = dataset.train_data
        test_callback = None
        if dataset.validation_data is None:
            fit_config['validation_data'] = dataset.test_data
        else:
            fit_config['validation_data'] = dataset.validation_data
            test_callback = AdditionalValidationSets(
                [(test_history_key, dataset.test_data)],
                batch_size=self.task.fit_config['batch_size'],
            )
            callbacks.append(test_callback)

        history = model.keras_model.fit(
            callbacks=callbacks,
            **fit_config,
        ).history  # type: ignore

        # Add test set history into history dict.
        if test_callback is not None:
            history.update(test_callback.history)
        return history

    def _make_result_record(
        self,
        dataset: Dataset,
        model: ModelInfo,
        history: Dict[str, Any],
    ) -> Dict[str, Any]:
        parameters: Dict[str, Any] = self.task.get_parameters()
        parameters.update(self.worker.worker_info)
        parameters.update(model.network.description)
        parameters.update({
            'num_free_parameters': model.network.num_free_parameters,
            # 'output_activation': network.output_activation,
            'network_structure': \
                jobqueue_marshal.marshal(model.network.structure),
            'python_version': str(platform.python_version()),
            'platform': str(platform.platform()),
            'tensorflow_version': str(tensorflow.__version__),
            'hostname': str(platform.node()),
            'slurm_job_id': os.getenv("SLURM_JOB_ID"),
        })

        git_hash = None
        try:
            git_hash = subprocess.check_output(
                ["git", "describe", "--always"],
                cwd=os.path.dirname(__file__)).strip().decode()
        except:
            pass
        parameters['git_hash'] = git_hash

        # TODO: migrate / add these to existing:
        parameters.update({
            'input_shape': dataset.input_shape,
            'output_shape': dataset.output_shape,
            'ml_task': dataset.ml_task,
            # 'training_set_size' : dataset.train_data.shape[0],
            # 'test_set_size' : dataset.train_data.shape[0],
            # 'validation_set_size' : dataset.train_data.shape[0],
        })

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
        parameters.update(history)
        return parameters