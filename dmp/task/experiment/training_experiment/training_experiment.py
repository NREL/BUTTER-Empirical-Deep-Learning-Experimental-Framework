from __future__ import annotations
import os
from dataclasses import dataclass
import itertools
import random
from typing import Any, Dict, Iterable, Optional, Set, Type
import numpy
import tensorflow
import tensorflow.keras as keras
from dmp import common
from dmp.common import KerasConfig
from dmp.model.network_info import NetworkInfo
from dmp.parquet_util import make_dataframe_from_dict
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.model_saving.model_saving_callback import ModelSavingCallback
from dmp.task.experiment.pruning_experiment.count_masked_parameters import (
    count_masked_parameters,
)
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder
from dmp.task.experiment.training_experiment import (
    training_experiment_keys,
    training_experiment_summarizer,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec

from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


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
from dmp.task.experiment.experiment import Experiment
from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.model.model_info import ModelInfo

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.context import Context


@dataclass
class TrainingExperiment(Experiment):
    # floating point precision {'float16', 'float32', 'float64'}

    precision: str

    dataset: DatasetSpec  # what dataset to train on

    model: ModelSpec  # defines network

    # contains batch size, epochs, shuffle
    fit: KerasConfig

    # contains learning rate
    optimizer: KerasConfig

    # The loss function to use. Set to None for runtime determination.
    loss: Optional[KerasConfig]

    # keras config for early stopping callback
    early_stopping: Optional[KerasConfig]

    # defines important strings and collections of strings for this experiment class
    keys = training_experiment_keys.keys

    # used to aggregate and summarize repetitions
    summarizer = training_experiment_summarizer.summarizer

    @property
    def version(self) -> int:
        return super().version + 20

    def __call__(
        self,
        context: Context,
        run: RunSpec,
    ) -> None:
        # tensorflow.config.optimizer.set_jit(True)
        self._setup_environment(run)
        dataset, metrics = self._load_and_prepare_dataset()
        network = self._make_network(self.model)
        model = self._make_model_from_network(network, metrics)
        experiment_history = self._try_restore_checkpoint(context, run, model)
        print(model.network.structure.summary())
        model.keras_model.summary()

        """
        + for resumable checkpointing:
            + always:
                + save checkpoint (weights, optimizer, epoch numbers)
                + experiment history
            + some cases:
                + model structure
                + experiment state
                    + maybe overwrite / new Task
            + save_checkpoint function passed to fit
                + saves these things in a way that can be resumed
            -> so:
                + save checkpoint
                    + to disk
                    + to db
                + save history to db
                + update Job & Task to resume on failure
                + add attr to track job/task execution history?

                + when running, must be able to resume
        """

        self._fit_model(
            context,
            run,
            self.fit,
            dataset,
            model,
            [self._make_early_stopping_callback()],
            experiment_history=experiment_history,
            new_seed=False,
        )

        self._record_result(
            context,
            run,
            dataset,
            model.network,
            experiment_history,
        )

    @classmethod
    def summarize(
        cls: Type["TrainingExperiment"],
        results: Sequence[ExperimentResultRecord],
    ) -> ExperimentSummaryRecord:
        return cls.summarizer.summarize(cls, results)

    def _setup_environment(self, run: RunSpec) -> None:
        seed = run.seed
        os.environ["PYTHONHASHSEED"] = str(seed)
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
            self.fit["batch_size"],
        )
        metrics = self._configute_for_dataset(dataset)
        return dataset, metrics

    def _configute_for_dataset(
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
        output_kernel_initializer = "HeUniform"
        output_activation = "relu"
        loss = "MeanSquaredError"
        if ml_task == MLTask.regression:
            output_activation = "sigmoid"
            output_kernel_initializer = "GlorotUniform"
            loss = "MeanSquaredError"
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
                output_activation = "sigmoid"
                output_kernel_initializer = "GlorotUniform"
                loss = "BinaryCrossentropy"
                metrics.extend(
                    [
                        keras.metrics.BinaryCrossentropy(),
                        "accuracy",
                        keras.metrics.Hinge(),
                        keras.metrics.SquaredHinge(),
                        keras.metrics.Precision(),
                        keras.metrics.Recall(),
                        keras.metrics.AUC(),
                    ]
                )
            else:
                output_activation = "softmax"
                output_kernel_initializer = "GlorotUniform"
                loss = "CategoricalCrossentropy"
                metrics.extend(
                    [
                        keras.metrics.CategoricalCrossentropy(),
                        "accuracy",
                        keras.metrics.CategoricalHinge(),
                    ]
                )
        else:
            raise Exception('Unknown task "{}"'.format(ml_task))

        model = self.model
        if model.input is None:
            model.input = Input()
        if model.input.get("shape", None) is None:
            input_shape = dataset.input_shape
            model.input["shape"] = input_shape

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
                    "activation": None,
                    "kernel_initializer": None,
                },
            )

        output = model.output
        if isinstance(output, Dense):
            if output.get("units", None) is None:
                output["units"] = int(dataset.output_shape[0])
            if output.get("activation", None) is None:
                # output['activation'] = make_keras_config(output_activation)
                output["activation"] = output_activation
            if output.get("kernel_initializer", None) is None:
                output["kernel_initializer"] = make_keras_config(
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

        if self.precision in {"mixed_float16", "mixed_bfloat16"}:
            keras.backend.set_floatx("float32")
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

    def _try_restore_checkpoint(
        self,
        context: Context,
        run: RunSpec,
        model: ModelInfo,
    ) -> Dict[str, Any]:
        checkpoint = run.resume_checkpoint
        if checkpoint is None:
            return {}
        optimizer = model.keras_model.optimizer
        grad_vars = model.keras_model.trainable_weights
        zero_grads = [tensorflow.zeros_like(w) for w in grad_vars]
        optimizer.apply_gradients(zip(zero_grads, grad_vars))
        checkpoint.resume(model)
        run_history = context.schema.get_run_history(checkpoint.run_id)
        if run_history is None:
            return {}
        run_history = run_history[run_history["epoch"] <= checkpoint.epoch]
        return run_history.to_dict(orient="list")  # type: ignore

    def _fit_model(
        self,
        context: Context,
        run: RunSpec,
        fit_config: Dict[str, Any],
        dataset: PreparedDataset,
        model: ModelInfo,
        callbacks: List[Optional[keras.callbacks.Callback]],
        new_model_number: bool = True,
        epochs: Optional[int] = None,
        experiment_history: Optional[Dict[str, Any]] = None,
        new_seed=False,
    ) -> Dict[str, Any]:
        # filter None's out of callbacks
        callbacks = [cb for cb in callbacks if cb is not None]

        # setup training, validation, and test datasets
        fit_config = fit_config.copy()
        fit_config["x"] = dataset.train
        fit_config["validation_data"] = dataset.validation

        if epochs is not None:
            fit_config["epochs"] = epochs

        keys = self.keys

        test_set_info = TestSetInfo(keys.test, dataset.test)
        validation_set_info = TestSetInfo(keys.validation, dataset.validation)
        train_set_info = TestSetInfo(keys.train, dataset.train)

        # setup model saving callback
        self._setup_model_saving_callback(
            context,
            run,
            callbacks,
            experiment_history,
            model,
        )

        # setup history statistics recorders
        timestamp_recorder = (
            TimestampRecorder(
                "_" + keys.interval_suffix,
                keys.epoch_start_time_ms,
                keys.epoch_time_ms,
            )
            if run.record_times
            else None
        )
        zero_epoch_recorder = ZeroEpochRecorder(
            [train_set_info, validation_set_info, test_set_info], None
        )

        additional_test_sets = [test_set_info]
        if run.record_post_training_metrics:
            additional_test_sets.append(TestSetInfo(keys.trained, dataset.train))

        history_callbacks = [
            timestamp_recorder,
            zero_epoch_recorder,
            TestSetHistoryRecorder(additional_test_sets, timestamp_recorder),
        ]
        callbacks.extend(history_callbacks)

        # fit the model
        history: keras.callbacks.History = model.keras_model.fit(
            callbacks=callbacks,
            verbose=0,  # type: ignore
            **fit_config,
        )  # type: ignore

        # convert keras History dictionary and epoch list to our standard
        self.remap_key_prefixes(
            history.history,
            [
                ("val_", keys.validation + "_", True),
                # (test_history_key + '_', 'test_'),
                ("", keys.train + "_", True),
            ],
        )
        history_callbacks.append(history)

        # copy zero epoch recorder's train_ metrics to trained_ metrics
        if run.record_post_training_metrics:
            self.remap_key_prefixes(
                zero_epoch_recorder.history,
                [
                    (keys.train + "_", keys.trained + "_", False),
                ],
            )

        # Add test set history into history dict.
        fit_history = self._merge_callback_histories(history_callbacks)

        # set free parameter count history
        fit_history_length = len(fit_history[keys.epoch])
        if keys.free_parameter_count_key not in fit_history:
            fit_history[keys.free_parameter_count_key] = [
                model.network.num_free_parameters
            ] * fit_history_length

        # set masked parameter count history
        if keys.maked_parameter_count_key not in fit_history:
            fit_history[keys.maked_parameter_count_key] = [
                count_masked_parameters(
                    model.network.structure,
                    model.keras_network.layer_to_keras_map,
                )
            ] * fit_history_length

        # set retained column
        if keys.retained not in fit_history:
            retained = [True] * fit_history_length
            fit_history[keys.retained] = retained

            early_stopping_callback = next(
                (
                    cb
                    for cb in callbacks
                    if isinstance(cb, keras.callbacks.EarlyStopping)
                ),
                None,
            )

            if (
                early_stopping_callback is not None
                and early_stopping_callback.stopped_epoch > 0
            ):
                last_retained_epoch = (
                    len(fit_history[keys.epoch]) - early_stopping_callback.patience
                )
                for i in range(last_retained_epoch + 1, fit_history_length):
                    retained[i] = False

        # if experiment_history was supplied, merge this call to fit into it and return it
        # if experiment_history is not None:
        return self._append_fit_history_to_model_history(
            new_model_number,
            new_seed,
            experiment_history,
            fit_history,
        )

        # return fit_history

    @staticmethod
    def remap_key_prefixes(
        target: Dict[str, Any],
        prefix_mapping: Iterable[Tuple[str, str, bool]],
    ) -> dict:
        plan = []
        for k, v in target.items():
            for src_prefix, dst_prefix, rename in prefix_mapping:
                if k.startswith(src_prefix):
                    plan.append((k, v, dst_prefix + k[len(src_prefix) :], rename))
                    break

        for src_key, v, dst_key, rename in plan:
            if rename:
                del target[src_key]

        for src_key, v, dst_key, rename in plan:
            target[dst_key] = v
        return target

    def _merge_callback_histories(
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

    def get_last_value_of(
        self,
        history: Dict[str, Any],
        key: str,
        default_value: Any,
    ) -> Any:
        vals = history.get(key, [])
        if len(vals) > 0:
            return vals[-1]
        return default_value

    def get_current_epoch(
        self,
        experiment_history: Optional[Dict[str, Any]],
    ) -> TrainingEpoch:
        if self.resume_from is None:
            initial_epoch = TrainingEpoch(0, -1, 0)
        else:
            initial_epoch = self.resume_from.get_epoch()

        if experiment_history is None:
            return initial_epoch

        history_epoch = TrainingEpoch(
            self.get_last_value_of(
                experiment_history,
                self.keys.epoch,
                0,
            ),
            self.get_last_value_of(
                experiment_history,
                self.keys.model_number,
                0,
            ),
            self.get_last_value_of(
                experiment_history,
                self.keys.model_epoch,
                0,
            ),
        )

        return max(initial_epoch, history_epoch)

    def _append_fit_history_to_model_history(
        self,
        new_model_number: bool,
        new_seed: bool,
        experiment_history: Optional[Dict[str, Any]],
        fit_history: Dict[str, Any],
    ) -> Dict[str, Any]:
        epoch = self.get_current_epoch(experiment_history)

        if new_model_number:
            epoch.count_new_model()

        fit_history_length = len(fit_history[self.keys.epoch])

        # set model number column
        fit_history[self.keys.model_number] = [epoch.model_number] * fit_history_length

        # set model epoch column
        model_epochs = numpy.array(fit_history[self.keys.epoch])
        model_epochs += epoch.model_epoch
        fit_history[self.keys.model_epoch] = model_epochs

        # convert model epochs to history epochs
        fit_history[self.keys.epoch] = model_epochs + epoch.epoch

        # set seed number column
        if self.keys.seed_number not in fit_history:
            seed_number = (
                self.get_last_value_of(fit_history, self.keys.seed_number, 0) + new_seed
            )
            fit_history[self.keys.seed_number] = [seed_number] * fit_history_length

        if experiment_history is None:
            return fit_history

        experiment_history_length = len(experiment_history.get(self.keys.epoch, []))
        metrics = set(itertools.chain(experiment_history.keys(), fit_history.keys()))
        for metric in metrics:
            if metric not in experiment_history:
                experiment_history[metric] = [None] * experiment_history_length
            if metric not in fit_history:
                fit_history[metric] = [None] * fit_history_length
            experiment_history[metric].extend(fit_history[metric])

        return experiment_history

    def _record_result(
        self,
        context: Context,
        run: RunSpec,
        dataset: PreparedDataset,
        network: NetworkInfo,
        history: Dict[str, Any],
    ) -> None:
        import platform

        # update run data
        run.data.update(
            {
                "job_id": context.id,
                "run_id": context.id,
                "python_version": str(platform.python_version()),
                "platform": str(platform.platform()),
                "tensorflow_version": str(tensorflow.__version__),
                "host_name": str(platform.node()),
                "slurm_job_id": common.get_slurm_job_id(),
                "git_hash": common.get_git_hash(),
                "context": context.info,
            }
        )

        # update experiment data
        self.data.update(
            {
                "ml_task": dataset.ml_task.value,
                "num_free_parameters": network.num_free_parameters,
                # 'model_structure':
                # network.structure,
                "input_shape": dataset.input_shape,
                "output_shape": dataset.output_shape,
                "train_set_size": dataset.train_size,
                "test_set_size": dataset.test_size,
                "validation_set_size": dataset.validation_size,
                "data_set_size": dataset.train_size
                + dataset.test_size
                + dataset.validation_size,
                "network_description": network.description,
            }
        )

        extended_history = self._extract_extended_history(history)

        context.record_result(
            self,
            run,
            history,
            extended_history,
        )
        # return ExperimentResultRecord(
        #     experiment_attrs,
        #     experiment_tags,
        #     run_data,
        #     make_dataframe_from_dict(experiment_history),
        #     None
        #     if len(extended_history) == 0
        #     else make_dataframe_from_dict(extended_history),  # type: ignore
        # )

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

    def _setup_model_saving_callback(
        self,
        context: Context,
        run: RunSpec,
        callbacks: List[Optional[keras.callbacks.Callback]],
        experiment_history: Optional[Dict[str, Any]],
        model: ModelInfo,
    ) -> Optional[keras.callbacks.Callback]:
        if run.model_saving is None:
            return None

        model_saving_callback = next(
            (cb for cb in callbacks if isinstance(cb, ModelSavingCallback)),
            None,
        )

        if model_saving_callback is not None:
            return model_saving_callback

        # if no model saving callback, create one
        model_saving_callback = run.model_saving.make_save_model_callback(
            context,
            self.get_current_epoch(experiment_history),
            model,
        )
        callbacks.append(model_saving_callback)
        return model_saving_callback

    def _save_checkpoint(
        self,
        context: Context,
        run: RunSpec,
        model: ModelInfo,
        history: Dict[str, Any],
    ) -> TrainingExperimentCheckpoint:
        # + save checkpoint
        #     + to disk
        #     + to db
        self.resume_from = context.save_model(
            model,
            epoch,
        )  # TODO

        # + save history to db

        # + update Job & Task to resume on failure
        # + add table to track job/task execution history?
        context.checkpoint_task(model, epoch, history)

        return self.resume_from
