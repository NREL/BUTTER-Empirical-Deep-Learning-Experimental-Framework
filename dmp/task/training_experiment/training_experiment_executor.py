from copy import deepcopy
import random
from typing import Any, Dict
import os
import platform
import subprocess
import tensorflow
import tensorflow.keras as keras
import numpy

from dmp.jobqueue_interface import jobqueue_marshal
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
        model = self._make_model(self.task.model)
        self._compile_model(dataset, model)
        callbacks = self._make_callbacks()
        fit_config = deepcopy(self.task.fit_config)
        history = self._fit_model(fit_config, dataset, model, callbacks)
        return self._make_result_record(dataset, model, history)

    def _set_random_seeds(self) -> None:
        seed: int = self.task.seed
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

    def _load_and_prepare_dataset(self) -> Dataset:
        dataset = Dataset.make(
            self.task.dataset,
            self.task.fit_config['batch_size'],
        )

        # auto-populate model inputs and outputs if not already set
        output_activation, loss = \
            get_output_activation_and_loss_for_ml_task(dataset)

        if len(self.task.model.inputs) == 0:
            self.task.model.inputs = \
                [Input({'shape': dataset.input_shape}, [])]

        if len(self.task.model.outputs) == 0:
            self.task.model.outputs = [
                Dense(
                    {
                        'units': dataset.output_shape[0],
                        'activation': output_activation,
                    }, [])
            ]

        return dataset

    def _make_model(self, model_spec: ModelSpec) -> ModelInfo:
        return self._make_model_from_network(self._make_network(model_spec))

    def _make_network(self, model_spec: ModelSpec) -> NetworkInfo:
        return model_spec.make_network()

    def _make_model_from_network(self, network: NetworkInfo):
        with worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)
            return make_keras_model_from_network(network)

    def _compile_model(self, dataset: Dataset, model: ModelInfo) -> None:
        if self.task.loss is None:
            _, self.task.loss = get_output_activation_and_loss_for_ml_task(
                dataset)

        model.keras_model.compile(
            loss=make_from_config_using_keras_get(
                self.task.loss,
                keras.losses.get,
                'loss',
            ),
            optimizer=make_from_config_using_keras_get(
                self.task.optimizer,
                keras.optimizers.get,
                'optimizer',
            ),  # type: ignore
            metrics=[
                'accuracy',
                keras.metrics.CosineSimilarity(),
                keras.metrics.Hinge(),
                keras.metrics.KLDivergence(),
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.MeanSquaredLogarithmicError(),
                keras.metrics.RootMeanSquaredError(),
                keras.metrics.SquaredHinge(),
            ],
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
        parameters: Dict[str, Any] = self.task.parameters
        parameters.update(self.worker.worker_info)
        parameters.update(model.network.description)
        parameters.update({
            'num_free_parameters': model.network.num_free_parameters,
            # 'output_activation': network.output_activation,
            'structure': \
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

    def _make_keras_loss(self) -> keras.losses.Loss:
        return make_from_config_using_keras_get(
            self.task.loss,
            keras.losses.get,
        )  # type: ignore
