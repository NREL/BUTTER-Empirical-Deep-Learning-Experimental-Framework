from dataclasses import dataclass
import random
from typing import Any, Dict, Optional
import os
import platform
import subprocess
import uuid
from jobqueue.job import Job
import tensorflow
import tensorflow.keras as keras
import numpy
import pyarrow

from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_config
from dmp.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.layer import *
from dmp.task.recorder.timestamp_recorder import TimestampRecorder
from dmp.task.task_result_record import TaskResultRecord
from dmp.task.recorder.test_set_history_recorder import TestSetHistoryRecorder
from dmp.task.task_util import *
from dmp.task.training_experiment.test_set_info import TestSetInfo
from dmp.task.recorder.test_set_recorder import TestSetRecorder
from dmp.model.network_info import NetworkInfo
from dmp.model.model_info import ModelInfo
from dmp.task.recorder.zero_epoch_recorder import ZeroEpochRecorder

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec
from dmp.task.task import Task
from dmp.task.training_experiment.training_experiment_keys import TrainingExperimentKeys
from dmp.worker import Worker


@dataclass
class TrainingExperiment(Task):
    precision: str  # floating point precision {'float16', 'float32', 'float64'}
    dataset: DatasetSpec  # migrate dataset stuff into here
    model: ModelSpec  # defines network
    fit: dict  # contains batch size, epochs, shuffle (migrate from run_config)
    optimizer: dict  # contains learning rate (migrate converting to typed config from keras serialization)
    loss: Optional[dict]  # set to None for runtime determination
    early_stopping: Optional[dict]  # direct migration

    record_post_training_metrics: bool  # new default false
    record_times: bool
    record_model: Optional[Any]
    record_metrics: Optional[Any]

    key_names = TrainingExperimentKeys()

    @property
    def version(self) -> int:
        return 10

    def __call__(self, worker: Worker, job: Job, *args,
                 **kwargs) -> TaskResultRecord:
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

    def _set_random_seeds(self) -> None:
        seed: int = self.seed
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

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

    def _make_model(self, worker, model_spec: ModelSpec) -> ModelInfo:
        return self._make_model_from_network(
            worker,
            self._make_network(model_spec),
        )

    def _make_network(self, model_spec: ModelSpec) -> NetworkInfo:
        return model_spec.make_network()

    def _make_model_from_network(self, worker, network: NetworkInfo):
        if self.precision in {'mixed_float16', 'mixed_bfloat16'}:
            keras.backend.set_floatx('float32')
            keras.mixed_precision.set_global_policy(
                keras.mixed_precision.Policy(self.precision))
        else:
            tensorflow.keras.backend.set_floatx(self.precision)
        with worker.strategy.scope() as s:  # type: ignore
            # tensorflow.config.optimizer.set_jit(True)
            return make_keras_model_from_network(network)

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
            self, fit_config: Dict[str, Any], dataset: PreparedDataset,
            model: ModelInfo,
            callbacks: List[Optional[keras.callbacks.Callback]]) -> Dict:
        callbacks = [cb for cb in callbacks if cb is not None]

        # setup training, validation, and test datasets
        fit_config = fit_config.copy()
        fit_config['x'] = dataset.train
        fit_config['validation_data'] = dataset.validation

        test_set_info = TestSetInfo(self.key_names.test, dataset.test)
        validation_set_info = TestSetInfo(self.key_names.validation,
                                          dataset.validation)
        train_set_info = TestSetInfo(self.key_names.train, dataset.train)

        timestamp_recorder = TimestampRecorder() if self.record_times else None
        zero_epoch_recorder = ZeroEpochRecorder(
            [train_set_info, validation_set_info, test_set_info], None)

        additional_test_sets = [test_set_info]
        if self.record_post_training_metrics:
            additional_test_sets.append(
                TestSetInfo(self.key_names.trained, dataset.train))

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
        remap_key_prefixes(
            history.history,
            [
                ('val_', self.key_names.validation + '_', True),
                # (test_history_key + '_', 'test_'),
                ('', self.key_names.train + '_', True),
            ])
        history_callbacks.append(history)

        if self.record_post_training_metrics:
            # copy zero epoch recorder's train_ metrics to trained_ metrics
            remap_key_prefixes(zero_epoch_recorder.history, [
                (self.key_names.train + '_', self.key_names.trained + '_',
                 False),
            ])

        # Add test set history into history dict.
        return self._merge_histories(history_callbacks)

    def _make_callbacks(self) -> List[Optional[keras.callbacks.Callback]]:
        return [self._make_early_stopping_callback()]

    def _make_early_stopping_callback(
            self) -> Optional[keras.callbacks.EarlyStopping]:
        return make_keras_instance(self.early_stopping)

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
        merged_history = {self.key_names.epoch: epochs}
        for metric, epoch_map in metric_map.items():
            merged_history[metric] = [
                epoch_map.get(epoch, None) for epoch in epochs
            ]
        return merged_history

    def _append_history_dicts(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any],
    ) -> None:
        for metric, metric_history in source.items():
            target.setdefault(metric, []).extend(metric_history)

    def _make_result_record(
        self,
        worker_info: Dict[str, Any],
        job_id: uuid.UUID,
        dataset: PreparedDataset,
        network: NetworkInfo,
        history: Dict[str, Any],
    ) -> TaskResultRecord:

        experiment_parameters = self.get_parameters()
        experiment_parameters.update({
            'ml_task': dataset.ml_task.value,
            'num_free_parameters':
            network.num_free_parameters,
            # 'model_structure':
            # network.structure,
            'input_shape':
            dataset.input_shape,
            'output_shape':
            dataset.output_shape,
            'train_set_size':
            dataset.train_size,
            'test_set_size':
            dataset.test_size,
            'validation_set_size':
            dataset.validation_size,
            'data_set_size':
            dataset.train_size + dataset.test_size + dataset.validation_size
        })

        for k, v in network.description.items():
            experiment_parameters[f'model_{k}'] = v

        run_data = {
            'job_id': job_id,
            'run_id': job_id,
            'python_version': str(platform.python_version()),
            'platform': str(platform.platform()),
            'tensorflow_version': str(tensorflow.__version__),
            'host_name': str(platform.node()),
            'slurm_job_id': self._get_slurm_id(),
            'git_hash': self._get_git_hash(),
        }

        run_data.update(worker_info)

        for key in ('seed', 'precision', 'batch', 'task_version',
                    'record_post_training_metrics', 'record_times',
                    'record_model', 'record_metrics'):
            run_data[key] = experiment_parameters.pop(key, None)

        return TaskResultRecord(
            experiment_parameters,
            run_data,
            history,
        )

    @staticmethod
    def make_summary_record(experiment_data: pyarrow.Table):
        # median and iqr of various stats
        pass

    @staticmethod
    def get_per_epoch_statistics(
        experiment_data: pyarrow.Table,
        group: str,
        columns: List[str],
    ):

        quantiles = [0, .25, .5, .75, 1]
        quantile_options = pyarrow.compute.QuantileOptions(quantiles)

        aggregation_ops = []
        aggregation_ops.append((
            columns[0],
            'count',
            pyarrow.compute.CountOptions(mode='all'),
        ))

        for column in columns:
            aggregation_ops.append((column, 'count'))
            aggregation_ops.append((column, 'mean'))
            aggregation_ops.append((column, 'stddev'))
            aggregation_ops.append((column, 'quantile', quantile_options))

        experiment_data.group_by(group).aggregate(aggregation_ops)

    # def _map_history(
    #     self,
    #     dataset: PreparedDataset,
    #     history: Dict[str, Any],
    # ) -> Dict[str, Any]:
    #     return remap_key_prefixes(
    #         history,
    #         [
    #             ('val_', validation_key + '_'),
    #             # (test_history_key + '_', 'test_'),
    #             ('', train_key + '_'),
    #         ])  # type: ignore

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
        + XXX loss (migrate from runtime calculation)
            XXX output_activation, loss = get_output_activation_and_loss_for_ml_task(
            XXX dataset.output_shape[1], dataset.ml_task)
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