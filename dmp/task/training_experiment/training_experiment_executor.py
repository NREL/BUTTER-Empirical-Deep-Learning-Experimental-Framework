from copy import deepcopy
import random
from typing import Any, Dict
import os
import platform
import subprocess
import uuid
from prometheus_client import Metric
import tensorflow
import tensorflow.keras as keras
import numpy
from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset

from dmp.layer.visitor.keras_interface.keras_utils import make_keras_instance, make_keras_config
from dmp.layer.visitor.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.layer import *
from dmp.model.model_spec import ModelSpec
from dmp.task.task_result_record import TaskResultRecord
from dmp.task.training_experiment.validation_history_recorder import ValidationHistoryRecorder
from dmp.task.task_util import *
from dmp.task.training_experiment.training_experiment import TrainingExperiment
from dmp.model.network_info import NetworkInfo
from dmp.model.model_info import ModelInfo

test_history_key: str = 'test'


class TrainingExperimentExecutor():

    def __init__(self, task: TrainingExperiment, worker) -> None:
        self.task: TrainingExperiment = task
        self.worker = worker

    def __call__(self) -> TaskResultRecord:
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

    def _load_and_prepare_dataset(self) -> PreparedDataset:
        return PreparedDataset(
            self.task.dataset,
            self.task.fit_config['batch_size'],
        )

    def _autoconfigure_for_dataset(
        self,
        dataset: PreparedDataset,
    ) -> List[Union[str, keras.metrics.Metric]]:
        # auto-populate model inputs and outputs if not already set
        num_outputs: int = dataset.output_shape[0]
        ml_task: MLTask = dataset.ml_task

        metrics = [
            keras.metrics.CosineSimilarity(),
            keras.metrics.KLDivergence(),
        ]
        output_kernel_initializer = 'HeUniform'
        output_activation = 'relu'
        if ml_task == MLTask.regression:
            output_activation = 'sigmoid'
            output_kernel_initializer = 'GlorotUniform'
            loss = 'MeanSquaredError'
            metrics.extend([
                keras.metrics.MeanSquaredError(),
                keras.metrics.RootMeanSquaredError(),
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredLogarithmicError(),
            ])
        elif ml_task == MLTask.classification:
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
            input_dim = len(dataset.input_shape)
            if input_dim <= 2:
                model.input['shape'] = dataset.input_shape
            elif input_dim == 3:
                model.input['shape'] = dataset.input_shape[0:2]
                model.input['channels'] = dataset.input_shape[2]
            else:
                raise NotImplementedError(
                    f'Unsupported input shape {dataset.input_shape}.')

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
        with self.worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)
            return make_keras_model_from_network(network)

    def _compile_model(
        self,
        dataset: PreparedDataset,
        model: ModelInfo,
        metrics: List[Union[str, keras.metrics.Metric]],
    ) -> None:
        model.keras_model.compile(
            loss=make_keras_instance(self.task.loss),  # type: ignore
            optimizer=make_keras_instance(self.task.optimizer),
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
        dataset: PreparedDataset,
        model: ModelInfo,
        callbacks: List[keras.callbacks.Callback],
    ) -> Dict:
        # setup training, validation, and test datasets
        fit_config = fit_config.copy()
        fit_config['x'] = dataset.train
        test_callback = None
        if dataset.validation is None:
            fit_config['validation_data'] = dataset.test
        else:
            fit_config['validation_data'] = dataset.validation
            test_callback = ValidationHistoryRecorder(
                [(test_history_key, dataset.test)],
                batch_size=self.task.fit_config['batch_size'],
            )
            callbacks.append(test_callback)

        history = model.keras_model.fit(
            callbacks=callbacks,
            **fit_config,
        ).history  # type: ignore
        history = self._map_history(dataset, history)

        # Add test set history into history dict.
        if test_callback is not None:
            history.update(test_callback.history)
        return history

    def _make_result_record(
        self,
        dataset: PreparedDataset,
        model: ModelInfo,
        history: Dict[str, Any],
    ) -> TaskResultRecord:
        from dmp.jobqueue_interface import jobqueue_marshal

        experiment_parameters = self.task.get_parameters()
        experiment_parameters.update({
            'ml_task': dataset.ml_task.value,
        })

        experiment_data = {
            'model_computed_num_free_parameters':
            model.network.num_free_parameters,
            'model_computed_network_structure':
            jobqueue_marshal.marshal(model.network.structure),
            'input_shape':
            dataset.input_shape,
            'output_shape':
            dataset.output_shape,
            'training_set_size':
            dataset.train_size,
            'test_set_size':
            dataset.test_size,
            'validation_set_size':
            dataset.validation_size,
            'total_dataset_size':
            dataset.train_size + dataset.test_size + dataset.validation_size
        }

        experiment_data.update({
            f'model_computed_{k}': v
            for k, v in model.network.description.items()
        })

        run_data = {}
        if self.worker is not None:
            run_data = self.worker.worker_info.copy()
        run_data.update({
            'run_id':
            str(uuid.uuid4()),
            'seed':
            experiment_parameters.pop('seed', None),
            'batch':
            experiment_parameters.pop('batch', None),
            'task_version':
            experiment_parameters.pop('task_version', None),
            'python_version':
            str(platform.python_version()),
            'platform':
            str(platform.platform()),
            'tensorflow_version':
            str(tensorflow.__version__),
            'host_name':
            str(platform.node()),
            'slurm_job_id':
            self._get_slurm_id(),
            'git_hash':
            self._get_git_hash(),
        })

        return TaskResultRecord(
            experiment_parameters,
            experiment_data,
            run_data,
            history,
        )

    def _map_history(
        self,
        dataset: PreparedDataset,
        history: Dict[str, Any],
    ) -> Dict[str, Any]:
        # rename 'val_' keys to 'test_' and un-prefixed history keys to 'train_'
        if dataset.validation is None:
            return remap_key_prefixes(history, [
                ('val_', 'test_'),
                ('', 'train_'),
            ])  # type: ignore
        else:
            return remap_key_prefixes(history, [
                ('val_', 'validation_'),
                (test_history_key + '_', 'test_'),
                ('', 'train_'),
            ])  # type: ignore

    def _get_slurm_id(self) -> Optional[int]:
        try:
            return int(os.getenv("SLURM_JOB_ID"))  # type: ignore
        except:
            return None

    def _get_git_hash(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "describe", "--always"],
                cwd=os.path.dirname(__file__)).strip().decode()
        except:
            return None