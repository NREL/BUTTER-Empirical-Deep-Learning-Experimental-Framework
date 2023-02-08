from dataclasses import dataclass
from operator import index
from typing import Any, Dict, Iterable, Optional, Set, Type
from jobqueue.job import Job
import tensorflow.keras as keras

from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_config
from dmp.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.layer import *
from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.recorder.test_set_history_recorder import TestSetHistoryRecorder
from dmp.task.experiment.training_experiment.a_training_experiment import ATrainingExperiment
from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.model.model_info import ModelInfo
from dmp.task.experiment.recorder.zero_epoch_recorder import ZeroEpochRecorder

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec

from dmp.worker import Worker


@dataclass
class TrainingExperiment(ATrainingExperiment):
    dataset: DatasetSpec  # migrate dataset stuff into here
    model: ModelSpec  # defines network
    fit: dict  # contains batch size, epochs, shuffle (migrate from run_config)
    optimizer: dict  # contains learning rate (migrate converting to typed config from keras serialization)
    loss: Optional[dict]  # set to None for runtime determination
    early_stopping: Optional[dict]  # direct migration

    @property
    def version(self) -> int:
        return 11

    def __call__(self, worker: Worker, job: Job, *args,
                 **kwargs) -> ExperimentResultRecord:
        self._set_random_seeds()
        dataset = self._load_and_prepare_dataset()
        metrics = self._autoconfigure_for_dataset(dataset)
        model = self._make_model(worker, self.model)
        self._compile_model(dataset, model, metrics)
        history = self._fit_model(
            self.fit,
            dataset,
            model,
            self._make_callbacks(),
        )
        return self._make_result_record(
            worker.worker_info,
            job.id,
            dataset,
            model.network,
            history,
        )

    def _load_and_prepare_dataset(self) -> PreparedDataset:
        return PreparedDataset(
            self.dataset,
            self.fit['batch_size'],
        )

    def _autoconfigure_for_dataset(
        self,
        dataset: PreparedDataset,
    ) -> List[Union[str, keras.metrics.Metric]]:
        # auto-populate model inputs and outputs if not already set
        num_outputs: int = int(dataset.output_shape[0])
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

        model = self.model
        if model.input is None:
            model.input = Input()
        if model.input.get('shape', None) is None:
            input_shape = dataset.input_shape
            input_dim = len(input_shape)
            if input_dim <= 2:
                model.input['shape'] = input_shape
            elif input_dim == 3:
                model.input['shape'] = list(input_shape[0:2])
                model.input['channels'] = input_shape[2]
            else:
                raise NotImplementedError(
                    f'Unsupported input shape {input_shape}.')

        if model.output is None:
            model.output = Dense.make(
                int(dataset.output_shape[0]),
                {
                    'activation': None,
                    'kernel_initializer': None,
                },
            )

        output = model.output
        if isinstance(output, Dense):
            if output.get('units', None) is None:
                output['units'] = int(dataset.output_shape[0])
            if output.get('activation', None) is None:
                output['activation'] = make_keras_config(output_activation)
            if output.get('kernel_initializer', None) is None:
                output['kernel_initializer'] = make_keras_config(
                    output_kernel_initializer)

        if self.loss is None:
            self.loss = make_keras_config(loss)

        return metrics

    def _compile_model(
        self,
        dataset: PreparedDataset,
        model: ModelInfo,
        metrics: List[Union[str, keras.metrics.Metric]],
    ) -> None:
        model.keras_model.compile(
            loss=make_keras_instance(self.loss),  # type: ignore
            optimizer=make_keras_instance(self.optimizer),
            metrics=metrics,
            run_eagerly=False,
        )

    def _fit_model(
        self,
        fit_config: Dict[str, Any],
        dataset: PreparedDataset,
        model: ModelInfo,
        callbacks: List[Optional[keras.callbacks.Callback]],
    ) -> Dict:
        callbacks = [cb for cb in callbacks if cb is not None]

        # setup training, validation, and test datasets
        fit_config = fit_config.copy()
        fit_config['x'] = dataset.train
        fit_config['validation_data'] = dataset.validation

        test_set_info = TestSetInfo(self.keys.test, dataset.test)
        validation_set_info = TestSetInfo(self.keys.validation,
                                          dataset.validation)
        train_set_info = TestSetInfo(self.keys.train, dataset.train)

        timestamp_recorder = TimestampRecorder(
            '_' + self.keys.interval_suffix,
            self.keys.epoch_start_time_ms,
            self.keys.epoch_time_ms,
        ) if self.record.times else None
        zero_epoch_recorder = ZeroEpochRecorder(
            [train_set_info, validation_set_info, test_set_info], None)

        additional_test_sets = [test_set_info]
        if self.record.post_training_metrics:
            additional_test_sets.append(
                TestSetInfo(self.keys.trained, dataset.train))

        history_callbacks = [
            timestamp_recorder,
            zero_epoch_recorder,
            TestSetHistoryRecorder(additional_test_sets, timestamp_recorder),
        ]

        callbacks.extend(history_callbacks)

        history: keras.callbacks.History = model.keras_model.fit(
            callbacks=callbacks,
            verbose=0,  # type: ignore
            **fit_config,
        )  # type: ignore

        # convert keras History dictionary and epoch list to our standard
        self.remap_key_prefixes(
            history.history,
            [
                ('val_', self.keys.validation + '_', True),
                # (test_history_key + '_', 'test_'),
                ('', self.keys.train + '_', True),
            ])
        history_callbacks.append(history)

        if self.record.post_training_metrics:
            # copy zero epoch recorder's train_ metrics to trained_ metrics
            self.remap_key_prefixes(zero_epoch_recorder.history, [
                (self.keys.train + '_', self.keys.trained + '_', False),
            ])

        # Add test set history into history dict.
        return self._merge_histories(history_callbacks)

    def _make_callbacks(self) -> List[Optional[keras.callbacks.Callback]]:
        return [self._make_early_stopping_callback()]

    def _make_early_stopping_callback(
            self) -> Optional[keras.callbacks.EarlyStopping]:
        return make_keras_instance(self.early_stopping)
