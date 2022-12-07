from typing import Any, Dict
import os
import platform
import subprocess
import tensorflow
import tensorflow.keras as keras
import numpy

from dmp.jobqueue_interface import jobqueue_marshal
from dmp.layer.visitor.keras_interface.layer_to_keras import KerasLayer, make_keras_network_from_layer
import dmp.task.aspect_test.keras_utils as keras_utils
from dmp.dataset.dataset import Dataset
from dmp.layer import *
from dmp.task.training_experiment.additional_validation_sets import AdditionalValidationSets
from dmp.task.task_util import *
from dmp.task.training_experiment.training_experiment import TrainingExperiment
from dmp.model.model_data import ModelData

test_history_key: str = 'test'


class TrainingExperimentExecutor():

    def __init__(self, task: TrainingExperiment, worker) -> None:
        self.task: TrainingExperiment = task
        self.worker = worker

    def __call__(self) -> Dict[str, Any]:
        self._set_random_seeds()
        dataset = self._load_and_prepare_dataset()
        model = self._make_model()
        self._compile_model(model)
        callbacks = self._make_callbacks()
        fit_config = deepcopy(self.task.fit_config)
        history = self._fit_model(fit_config, dataset, model, callbacks)
        return self._make_result_record(model, history)

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

    def _make_model(self) -> ModelData:
        task: TrainingExperiment = self.task

        (
            structure,
            num_free_parameters,
            layer_shapes,
        ) = task.model.make_network()

        # make a keras model from the network structure
        keras_model = None
        with worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)
            (
                keras_inputs,
                keras_outputs,
                layer_to_keras_map,
            ) = make_keras_network_from_layer(structure, layer_shapes)

            keras_model = keras.Model(
                inputs=keras_inputs,
                outputs=keras_outputs,
            )

            if len(keras_model.inputs) != 1:  # type: ignore
                raise ValueError('Wrong number of keras inputs generated')

        if keras_utils.count_trainable_parameters_in_keras_model(keras_model)\
            != num_free_parameters:
            raise RuntimeError('Wrong number of trainable parameters')

        return ModelData(
            structure,
            layer_shapes,
            widths,
            num_free_parameters,
            # output_activation,
            layer_to_keras_map,
            keras_model,  # type: ignore
        )

    def _compile_model(self, model: ModelData) -> None:
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
        network: ModelData,
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

        history = network.keras_model.fit(
            callbacks=callbacks,
            **fit_config,
        ).history  # type: ignore

        # Add test set history into history dict.
        if test_callback is not None:
            history.update(test_callback.history)
        return history

    def _make_result_record(
        self,
        model: ModelData,
        history: Dict[str, Any],
    ) -> Dict[str, Any]:
        parameters: Dict[str, Any] = self.task.parameters
        parameters.update(self.worker.worker_info)
        parameters.update({
            'widths': model.widths,
            'num_free_parameters': model.num_free_parameters,
            # 'output_activation': network.output_activation,
            'structure': \
                jobqueue_marshal.marshal(model.structure),
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
