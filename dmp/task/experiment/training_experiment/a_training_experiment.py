from numbers import Number
from dataclasses import dataclass
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
        if self.precision in {'mixed_float16', 'mixed_bfloat16'}:
            keras.backend.set_floatx('float32')
            keras.mixed_precision.set_global_policy(
                keras.mixed_precision.Policy(self.precision))
        else:
            tensorflow.keras.backend.set_floatx(self.precision)
        with worker.strategy.scope() as s:  # type: ignore
            # tensorflow.config.optimizer.set_jit(True)
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

    def _concatenate_histories(
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

        experiment_parameters = self.get_parameters()

        run_data_set = {'seed', 'precision', 'batch', 'task_version'}
        for key in list(experiment_parameters.keys()):
            if key in run_data_set or key.startswith('record_'):
                run_data[key] = experiment_parameters.pop(key, None)

        experiment_parameters.update({
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
            experiment_parameters[f'model_{k}'] = v

        extended_history = self._extract_extended_history(history)

        return ExperimentResultRecord(
            experiment_parameters,
            {},
            run_data,
            pandas.DataFrame(history),
            None if len(extended_history) == 0 else
            pandas.DataFrame(extended_history),
        )

    def _extract_extended_history(
        self,
        history: Dict[str, Union[List, numpy.ndarray]],
    ) -> Dict[str, Union[List, numpy.ndarray]]:
        extended_history = {}
        for column in self.keys.extended_history_columns:
            v = history.pop(column, None)
            if v is not None:
                extended_history[column] = v
        return extended_history