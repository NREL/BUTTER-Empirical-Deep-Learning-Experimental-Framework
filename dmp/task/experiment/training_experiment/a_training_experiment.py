from numbers import Number
from dataclasses import dataclass
from pprint import pprint
import random
from typing import Any, Dict, Iterable, Optional, Set, Type
import platform
import uuid
from jobqueue.job import Job
import pandas
import pandas.core.groupby.groupby
import tensorflow
import tensorflow.keras as keras
import numpy
from dmp import common

from dmp.dataset.prepared_dataset import PreparedDataset

from dmp.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.layer import *
from dmp.parquet_util import make_dataframe_from_dict
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.training_experiment import training_experiment_keys, training_experiment_summarizer
from dmp.task.experiment.training_experiment.experiment_record_settings import ExperimentRecordSettings
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder
from dmp.model.network_info import NetworkInfo
from dmp.model.model_info import ModelInfo

from dmp.model.model_spec import ModelSpec

from dmp.worker import Worker


@dataclass
class ATrainingExperiment(ExperimentTask):
    precision: str  # floating point precision {'float16', 'float32', 'float64'}
    record: ExperimentRecordSettings

    keys = training_experiment_keys.keys
    summarizer = training_experiment_summarizer.summarizer

    @classmethod
    def summarize(
        cls: Type['ATrainingExperiment'],
        results: Sequence[ExperimentResultRecord],
    ) -> ExperimentSummaryRecord:
        return cls.summarizer.summarize(cls, results)

    def _set_random_seeds(self) -> None:
        seed: int = self.seed
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

    def _make_model(
        self,
        worker: Worker,
        model_spec: ModelSpec,
    ) -> ModelInfo:
        return self._make_model_from_network(
            worker,
            self._make_network(model_spec),
        )

    def _make_network(self, model_spec: ModelSpec) -> NetworkInfo:
        return model_spec.make_network()

    def _make_model_from_network(
        self,
        worker: Worker,
        network: NetworkInfo,
    ):
        from dmp.marshaling import marshal
        pprint(marshal.marshal(network.structure))

        if self.precision in {'mixed_float16', 'mixed_bfloat16'}:
            keras.backend.set_floatx('float32')
            keras.mixed_precision.set_global_policy(
                keras.mixed_precision.Policy(self.precision))
        else:
            keras.backend.set_floatx(self.precision)

        return make_keras_model_from_network(network)

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
            merged_history[metric] = [
                epoch_map.get(epoch, None) for epoch in epochs
            ]
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

    def _accumulate_model_history(
        self,
        history: Dict[str, Any],
        model_history: Dict[str, Any],
        num_free_parameters: int,
        early_stopping_callback: Optional[Any],
    ) -> None:
        source_length = len(model_history[self.keys.epoch])

        # model number column
        model_history[self.keys.model_number] = [
            self._get_last_value_of(history, self.keys.model_number, -1) + 1
        ] * source_length

        # set model epoch column
        model_epochs = numpy.array(model_history[self.keys.epoch])
        model_history[self.keys.model_epoch] = model_epochs

        # convert model epochs to history epochs
        model_history[
            self.keys.epoch] = model_epochs + self._get_last_value_of(
                history, self.keys.epoch, 0)

        # free parameter count history
        model_history[self.keys.free_parameter_count_key] = [
            num_free_parameters
        ] * source_length

        # set retained column
        retained = [True] * source_length
        model_history[self.keys.retained] = retained

        if early_stopping_callback is not None and early_stopping_callback.stopped_epoch > 0:
            last_retained_epoch = len(model_history[
                self.keys.epoch]) - early_stopping_callback.patience
            for i in range(last_retained_epoch + 1, source_length):
                retained[i] = False

        self._extend_history(history, model_history)

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
        job_id: uuid.UUID,
        dataset: PreparedDataset,
        network: NetworkInfo,
        history: Dict[str, Any],
    ) -> ExperimentResultRecord:

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

        experiment_attrs = self.get_parameters()
        experiment_tags = {}

        run_data_set = {'seed', 'precision', 'task_version', 'batch'}
        tag_prefix = 'tags_'
        run_tags_prefix = 'run_tags_'
        for key in list(experiment_attrs.keys()):
            if key in run_data_set or key.startswith('record_'):
                run_data[key] = experiment_attrs.pop(key, None)
            elif key.startswith(tag_prefix):
                experiment_tags[key[len(tag_prefix):]] = experiment_attrs.pop(
                    key, None)
            elif key.startswith(run_tags_prefix):
                run_data[key[len(run_tags_prefix):]] = experiment_attrs.pop(
                    key, None)

        experiment_attrs.update({
            'ml_task':
            dataset.ml_task.value,
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
            experiment_attrs[f'model_{k}'] = v

        extended_history = self._extract_extended_history(history)

        return ExperimentResultRecord(
            experiment_attrs,
            experiment_tags,
            run_data,
            make_dataframe_from_dict(history),
            None if len(extended_history) == 0 else
            make_dataframe_from_dict(extended_history),
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