from dataclasses import dataclass
from operator import index
import random
from typing import Any, Dict, Iterable, Optional, Set, Type
from uuid import UUID
from jobqueue.job import Job
import numpy
import tensorflow
from dmp import common
from dmp.common import KerasConfig
from dmp.model.network_info import NetworkInfo
from dmp.parquet_util import make_dataframe_from_dict
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder
from dmp.task.experiment.training_experiment import (
    training_experiment_keys,
    training_experiment_summarizer,
)
from dmp.task.experiment.training_experiment.experiment_record_settings import (
    ExperimentRecordSettings,
)
from dmp.task.experiment.training_experiment.model_saving_callback import (
    ModelSavingCallback,
)
from dmp.task.experiment.training_experiment.resume_config import ResumeConfig
import tensorflow.keras as keras

from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_config
from dmp.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.layer import *
from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder
from dmp.task.experiment.recorder.test_set_history_recorder import (
    TestSetHistoryRecorder,
)
from dmp.task.experiment.recorder.zero_epoch_recorder import ZeroEpochRecorder
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.model.model_info import ModelInfo

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec

from dmp.worker import Worker


@dataclass
class TrainingExperiment(ExperimentTask):
    precision: str  # floating point precision {'float16', 'float32', 'float64'}
    record: ExperimentRecordSettings
    dataset: DatasetSpec  # migrate dataset stuff into here
    model: ModelSpec  # defines network
    fit: dict  # contains batch size, epochs, shuffle (migrate from run_config)
    optimizer: dict  # contains learning rate (migrate converting to typed config from keras serialization)
    loss: Optional[KerasConfig]  # set to None for runtime determination
    early_stopping: Optional[KerasConfig]  # keras config for early stopping callback
    resume_from: Optional[
        ResumeConfig
    ]  # resume this experiment from the supplied checkpoint

    keys = training_experiment_keys.keys
    summarizer = training_experiment_summarizer.summarizer

    @property
    def version(self) -> int:
        return 12

    def __call__(
        self, worker: Worker, job: Job, *args, **kwargs
    ) -> ExperimentResultRecord:
        with worker.strategy.scope():
            # tensorflow.config.optimizer.set_jit(True)
            self._set_random_seeds()
            dataset, metrics = self._load_and_prepare_dataset()
            network = self._make_network(self.model)
            model = self._make_model_from_network(network, metrics)
            self._resume_model(model)
            print(model.network.structure.summary())
            model.keras_model.summary()

            history = self._fit_model(
                worker,
                job,
                self.fit,
                dataset,
                model,
                [self._make_early_stopping_callback()],
            )
            return self._make_result_record(
                worker.worker_info,
                job.id,
                dataset,
                model.network,
                history,
            )

    @classmethod
    def summarize(
        cls: Type['TrainingExperiment'],
        results: Sequence[ExperimentResultRecord],
    ) -> ExperimentSummaryRecord:
        return cls.summarizer.summarize(cls, results)

    def _set_random_seeds(self) -> None:
        import os

        seed: int = self.seed        
        os.environ['PYTHONHASHSEED'] = str(seed)
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

        # NB: for strict TF determinisim:
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    def _make_network(self, model_spec: ModelSpec) -> NetworkInfo:
        return model_spec.make_network()

    def _load_and_prepare_dataset(
        self,
    ) -> Tuple[PreparedDataset, List[Union[str, keras.metrics.Metric]]]:
        dataset = PreparedDataset(
            self.dataset,
            self.fit['batch_size'],
        )
        metrics = self._autoconfigure_for_dataset(dataset)
        return dataset, metrics

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
        loss = 'MeanSquaredError'
        if ml_task == MLTask.regression:
            output_activation = 'sigmoid'
            output_kernel_initializer = 'GlorotUniform'
            loss = 'MeanSquaredError'
            metrics.extend(
                [
                    keras.metrics.MeanSquaredError(),
                    keras.metrics.RootMeanSquaredError(),
                    keras.metrics.MeanAbsoluteError(),
                    keras.metrics.MeanSquaredLogarithmicError(),
                ]
            )
        elif ml_task == MLTask.classification:
            if num_outputs == 1:
                output_activation = 'sigmoid'
                output_kernel_initializer = 'GlorotUniform'
                loss = 'BinaryCrossentropy'
                metrics.extend(
                    [
                        keras.metrics.BinaryCrossentropy(),
                        'accuracy',
                        keras.metrics.Hinge(),
                        keras.metrics.SquaredHinge(),
                        keras.metrics.Precision(),
                        keras.metrics.Recall(),
                        keras.metrics.AUC(),
                    ]
                )
            else:
                output_activation = 'softmax'
                output_kernel_initializer = 'GlorotUniform'
                loss = 'CategoricalCrossentropy'
                metrics.extend(
                    [
                        keras.metrics.CategoricalCrossentropy(),
                        'accuracy',
                        keras.metrics.CategoricalHinge(),
                    ]
                )
        else:
            raise Exception('Unknown task "{}"'.format(ml_task))

        model = self.model
        if model.input is None:
            model.input = Input()
        if model.input.get('shape', None) is None:
            input_shape = dataset.input_shape
            model.input['shape'] = input_shape

            # input_dim = len(input_shape)
            # print(f'input shape: {input_shape}')
            # if input_dim <= 2:
            #     model.input['shape'] = input_shape
            # elif input_dim == 3:
            #     # model.input['shape'] = list(input_shape[0:2])
            #     # model.input['filters'] = input_shape[2]
            # else:
            #     raise NotImplementedError(
            #         f'Unsupported input shape {input_shape}.')

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
                # output['activation'] = make_keras_config(output_activation)
                output['activation'] = output_activation
            if output.get('kernel_initializer', None) is None:
                output['kernel_initializer'] = make_keras_config(
                    output_kernel_initializer
                )

        if self.loss is None:
            self.loss = make_keras_config(loss)

        return metrics

    def _make_model_from_network(
        self,
        network: NetworkInfo,
        metrics: List[Union[str, keras.metrics.Metric]],
    ):
        # from dmp.marshaling import marshal
        # pprint(marshal.marshal(network.structure))
        print(network.structure.summary())

        if self.precision in {'mixed_float16', 'mixed_bfloat16'}:
            keras.backend.set_floatx('float32')
            keras.mixed_precision.set_global_policy(
                keras.mixed_precision.Policy(self.precision)
            )
        else:
            keras.backend.set_floatx(self.precision)

        model = make_keras_model_from_network(network)
        model.keras_model.compile(
            loss=make_keras_instance(self.loss),  # type: ignore
            optimizer=make_keras_instance(self.optimizer),
            metrics=metrics,
            run_eagerly=False,
            jit_compile=True,
        )
        return model

    def _resume_model(
        self,
        model: ModelInfo,
    ) -> None:
        if self.resume_from is not None:
            self.resume_from.resume(model)

    def _fit_model(
        self,
        worker: Worker,
        job: Job,
        fit_config: Dict[str, Any],
        dataset: PreparedDataset,
        model: ModelInfo,
        callbacks: List[Optional[keras.callbacks.Callback]],
        epochs: Optional[int] = None,
        experiment_history: Optional[Dict[str, Any]] = None,
        num_free_parameters: Optional[int] = None,
    ) -> Dict:
        # setup training, validation, and test datasets
        fit_config = fit_config.copy()
        fit_config['x'] = dataset.train
        fit_config['validation_data'] = dataset.validation

        if epochs is not None:
            fit_config['epochs'] = epochs

        test_set_info = TestSetInfo(self.keys.test, dataset.test)
        validation_set_info = TestSetInfo(self.keys.validation, dataset.validation)
        train_set_info = TestSetInfo(self.keys.train, dataset.train)

        timestamp_recorder = (
            TimestampRecorder(
                '_' + self.keys.interval_suffix,
                self.keys.epoch_start_time_ms,
                self.keys.epoch_time_ms,
            )
            if self.record.times
            else None
        )
        zero_epoch_recorder = ZeroEpochRecorder(
            [train_set_info, validation_set_info, test_set_info], None
        )

        additional_test_sets = [test_set_info]
        if self.record.post_training_metrics:
            additional_test_sets.append(TestSetInfo(self.keys.trained, dataset.train))

        history_callbacks = [
            timestamp_recorder,
            zero_epoch_recorder,
            TestSetHistoryRecorder(additional_test_sets, timestamp_recorder),
        ]

        model_saving = self._get_model_saving_callback(callbacks)
        if model_saving is None:
            # if no model saving callback, create one
            model_saving = self._make_model_saving_callback(
                worker,
                job,
            )
            callbacks.append(model_saving)

        if model_saving is not None:
            # configure model saving callback
            model_saving.model_info = model

        callbacks = [cb for cb in callbacks if cb is not None]

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
            ],
        )
        history_callbacks.append(history)

        if self.record.post_training_metrics:
            # copy zero epoch recorder's train_ metrics to trained_ metrics
            self.remap_key_prefixes(
                zero_epoch_recorder.history,
                [
                    (self.keys.train + '_', self.keys.trained + '_', False),
                ],
            )

        # Add test set history into history dict.
        run_history = self._merge_histories(history_callbacks)

        # if experiment_history was supplied, merge this call to fit into it and return it
        if experiment_history is not None:
            self._append_run_history_to_model_history(
                experiment_history,
                run_history,
                model.network.num_free_parameters
                if num_free_parameters is None
                else num_free_parameters,
                self._get_early_stopping_callback(callbacks),
            )

        return run_history

    def _merge_histories(
        self,
        histories: Iterable[Union[keras.callbacks.History, TestSetRecorder]],
    ) -> Dict[str, Any]:
        epoch_set = set()
        metric_map = {}

        for history in histories:
            for metric, metric_history in history.history.items():
                for epoch, value in zip(history.epoch, metric_history):
                    epoch += 1
                    epoch_set.add(epoch)
                    metric_map.setdefault(metric, {})[epoch] = value

        # offset epoch numbers by 1 (untrained network becomes the 0th epoch)
        epochs = sorted(epoch_set)
        merged_history = {self.keys.epoch: epochs}
        for metric, epoch_map in metric_map.items():
            merged_history[metric] = [epoch_map.get(epoch, None) for epoch in epochs]
        return merged_history

    def _get_last_epoch(
        self,
        history: Dict[str, Any],
    ) -> int:
        return self._get_last_value_of(history, self.keys.epoch, 0)

    def _get_last_value_of(
        self,
        history: Dict[str, Any],
        key: str,
        default_value: Any,
    ) -> Any:
        vals = history.get(key, [])
        if len(vals) > 0:
            return vals[-1]
        return default_value

    def _append_run_history_to_model_history(
        self,
        experiment_history: Dict[str, Any],
        model_history: Dict[str, Any],
        num_free_parameters: int,
        early_stopping_callback: Optional[Any],
    ) -> Dict[str, Any]:
        source_length = len(model_history[self.keys.epoch])

        default_last_epoch = 0
        default_last_model_number = 0
        default_last_model_epoch = 0
        if self.resume_from is not None:
            (
                default_last_epoch,
                default_last_model_number,
                default_last_model_epoch,
            ) = self.resume_from.get_epoch()

        # model number column
        model_number = (
            self._get_last_value_of(
                experiment_history,
                self.keys.model_number,
                default_last_model_number - 1,
            )
            + 1
        )
        model_history[self.keys.model_number] = [model_number] * source_length

        # set model epoch column
        model_epochs = numpy.array(model_history[self.keys.epoch])
        if model_number == default_last_model_number:
            model_epochs += default_last_model_epoch
        model_history[self.keys.model_epoch] = model_epochs

        # convert model epochs to history epochs
        model_history[self.keys.epoch] = model_epochs + self._get_last_value_of(
            experiment_history,
            self.keys.epoch,
            default_last_epoch,
        )

        # free parameter count history
        model_history[self.keys.free_parameter_count_key] = [
            num_free_parameters
        ] * source_length

        # set retained column
        retained = [True] * source_length
        model_history[self.keys.retained] = retained

        if (
            early_stopping_callback is not None
            and early_stopping_callback.stopped_epoch > 0
        ):
            last_retained_epoch = (
                len(model_history[self.keys.epoch]) - early_stopping_callback.patience
            )
            for i in range(last_retained_epoch + 1, source_length):
                retained[i] = False

        self._extend_history(experiment_history, model_history)
        return experiment_history

    def _extend_history(
        self,
        history: Dict[str, Any],
        additional_history: Dict[str, Any],
    ) -> None:
        for metric, metric_history in additional_history.items():
            history.setdefault(metric, []).extend(metric_history)

    def _make_result_record(
        self,
        worker_info: Dict[str, Any],
        job_id: UUID,
        dataset: PreparedDataset,
        network: NetworkInfo,
        history: Dict[str, Any],
    ) -> ExperimentResultRecord:
        import platform

        run_data = {
            'job_id': job_id,
            'run_id': job_id,
            'python_version': str(platform.python_version()),
            'platform': str(platform.platform()),
            'tensorflow_version': str(tensorflow.__version__),
            'host_name': str(platform.node()),
            'slurm_job_id': common.get_slurm_job_id(),
            'git_hash': common.get_git_hash(),
        }
        run_data.update(worker_info)

        experiment_attrs: Dict[str, Any] = self.get_parameters()
        experiment_tags = {}

        run_data_set = {'seed', 'precision', 'task_version', 'batch'}
        tag_prefix = 'tags_'
        run_tags_prefix = 'run_tags_'
        for key in list(experiment_attrs.keys()):
            if key in run_data_set or key.startswith('record_'):
                run_data[key] = experiment_attrs.pop(key, None)
            elif key.startswith(tag_prefix):
                experiment_tags[key[len(tag_prefix) :]] = experiment_attrs.pop(
                    key, None
                )
            elif key.startswith(run_tags_prefix):
                run_data[key[len(run_tags_prefix) :]] = experiment_attrs.pop(key, None)

        experiment_attrs.update(
            {
                'ml_task': dataset.ml_task.value,
                'num_free_parameters': network.num_free_parameters,
                # 'model_structure':
                # network.structure,
                'input_shape': dataset.input_shape,
                'output_shape': dataset.output_shape,
                'train_set_size': dataset.train_size,
                'test_set_size': dataset.test_size,
                'validation_set_size': dataset.validation_size,
                'data_set_size': dataset.train_size
                + dataset.test_size
                + dataset.validation_size,
            }
        )

        for k, v in network.description.items():
            experiment_attrs[f'model_{k}'] = v

        extended_history = self._extract_extended_history(history)

        return ExperimentResultRecord(
            experiment_attrs,
            experiment_tags,
            run_data,
            make_dataframe_from_dict(history),
            None
            if len(extended_history) == 0
            else make_dataframe_from_dict(extended_history),  # type: ignore
        )

    def _extract_extended_history(
        self,
        history: Dict[str, Union[List, numpy.ndarray]],
    ) -> Dict[str, Union[List, numpy.ndarray]]:
        keys = self.keys
        extended_history = {keys.epoch: history[keys.epoch]}
        for column in keys.extended_history_columns:
            v = history.pop(column, None)
            if v is not None:
                extended_history[column] = v
        return extended_history

    def _make_early_stopping_callback(self) -> Optional[keras.callbacks.EarlyStopping]:
        return make_keras_instance(self.early_stopping)

    def _make_model_saving_callback(
        self,
        worker: Worker,
        job: Job,
    ) -> Optional[keras.callbacks.Callback]:
        model_saving = self.record.model_saving
        if model_saving is None:
            return None
        return model_saving.make_save_model_callback(
            worker,
            job,
            self,
        )

    @staticmethod
    def _get_early_stopping_callback(
        callbacks: List[Optional[keras.callbacks.Callback]],
    ) -> Optional[keras.callbacks.EarlyStopping]:
        return next(
            (cb for cb in callbacks if isinstance(cb, keras.callbacks.EarlyStopping)),
            None,
        )

    @staticmethod
    def _get_model_saving_callback(
        callbacks: List[Optional[keras.callbacks.Callback]],
    ) -> Optional[ModelSavingCallback]:
        return next(
            (cb for cb in callbacks if isinstance(cb, ModelSavingCallback)),
            None,
        )
