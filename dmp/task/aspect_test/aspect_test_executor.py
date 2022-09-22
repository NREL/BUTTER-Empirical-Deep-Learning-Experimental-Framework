from dataclasses import dataclass
import json
import os
import platform
import subprocess
from typing import Any

from pytest import param

from dmp.data.pmlb import pmlb_loader
from dmp.jobqueue_interface.common import jobqueue_marshal

import tensorflow.keras.metrics as metrics
import tensorflow.keras.callbacks as callbacks

from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from dmp.task.aspect_test.aspect_test_utils import *
from dmp.task.task import Parameter

import pandas
import numpy

# from keras_buoy.models import ResumableModel

_datasets = pmlb_loader.load_dataset_index()


@dataclass
class AspectTestExecutor(AspectTestTask):
    '''
    '''

    output_activation: Optional[str] = None
    # tensorflow_strategy: Optional[tensorflow.distribute.Strategy] = None
    keras_model: Optional[tensorflow.keras.Model] = None
    run_loss: Optional[tensorflow.keras.losses.Loss] = None
    network_structure: Optional[NetworkModule] = None
    dataset_series: Optional[pandas.Series] = None
    inputs: Optional[numpy.ndarray] = None
    outputs: Optional[numpy.ndarray] = None

    def __call__(self, parent: AspectTestTask, worker, *args, **kwargs) \
            -> Dict[str, Any]:
        # # Configure hardware
        # if self.tensorflow_strategy is None:
        #     self.tensorflow_strategy = tensorflow.distribute.get_strategy()

        # Set random seeds
        self.seed = set_random_seeds(self.seed)

        # Load dataset
        self.dataset_series, self.inputs, self.outputs =  \
            pmlb_loader.load_dataset(_datasets, self.dataset)

        # prepare dataset shuffle, split, and label noise:
        prepared_config = prepare_dataset(
            self.test_split_method,
            self.test_split,
            self.label_noise,
            self.run_config,
            str(self.dataset_series['Task']),
            self.inputs,
            self.outputs,
        )

        # Generate neural network architecture
        num_outputs = self.outputs.shape[1]
        self.output_activation, self.run_loss = \
            compute_network_configuration(num_outputs, self.dataset_series)

        # TODO: make it so we don't need this hack
        shape = self.shape
        residual_mode = None
        residual_suffix = '_residual'
        if shape.endswith(residual_suffix):
            residual_mode = 'full'
            shape = shape[0:-len(residual_suffix)]

        layer_args = {
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
        }

        # Build NetworkModule network
        delta, widths, self.network_structure = find_best_layout_for_budget_and_depth(
            self.inputs,
            residual_mode,
            self.input_activation,
            self.activation,
            self.output_activation,
            self.size,
            widths_factory(shape)(num_outputs, self.depth),
            self.depth,
            shape,
            layer_args,
        )

        # reject non-conformant network sizes
        delta = count_num_free_parameters(self.network_structure) - self.size
        relative_error = delta / self.size
        if numpy.abs(relative_error) > .2:
            raise ValueError(
                f'Could not find conformant network error : {relative_error}%, delta : {delta}, size: {self.size}.')

        # Create and execute network using Keras
        with worker.strategy.scope() as s:  # type: ignore
            print(f'Tensorflow scope: {s}')
            # Build Keras model
            self.keras_model = make_keras_network_from_network_module(
                self.network_structure)
            if len(self.keras_model.inputs) != 1:  # type: ignore
                print(
                    f'weird error: {len(self.keras_model.inputs)}, {json.dumps(jobqueue_marshal.marshal(self.network_structure))}')  # type: ignore
                raise ValueError('Wrong number of keras inputs generated')

        # Compile Keras Model
        run_metrics = [
            # metrics.CategoricalAccuracy(),
            'accuracy',
            metrics.CosineSimilarity(),
            metrics.Hinge(),
            metrics.KLDivergence(),
            metrics.MeanAbsoluteError(),
            metrics.MeanSquaredError(),
            metrics.MeanSquaredLogarithmicError(),
            metrics.RootMeanSquaredError(),
            metrics.SquaredHinge(),
        ]

        run_optimizer = tensorflow.keras.optimizers.get(self.optimizer)

        self.keras_model.compile(
            # loss='binary_crossentropy', # binary classification
            # loss='categorical_crossentropy', # categorical classification (one hot)
            loss=self.run_loss,  # regression
            optimizer=run_optimizer,
            # optimizer='rmsprop',
            # metrics=['accuracy'],
            metrics=run_metrics,
            run_eagerly=False,
        )

        num_free_parameters = count_trainable_parameters_in_keras_model(
            self.keras_model)
        assert count_num_free_parameters(self.network_structure) == num_free_parameters, \
            'Wrong number of trainable parameters'

        # Configure Keras Callbacks
        run_callbacks = []
        if self.early_stopping is not None:
            run_callbacks.append(
                callbacks.EarlyStopping(**self.early_stopping))

        # # optionally enable checkpoints
        # if self.save_every_epochs is not None and self.save_every_epochs > 0:
        #     DMP_CHECKPOINT_DIR = os.getenv(
        #         'DMP_CHECKPOINT_DIR', default='checkpoints')
        #     if not os.path.exists(DMP_CHECKPOINT_DIR):
        #         os.makedirs(DMP_CHECKPOINT_DIR)

        #     save_path = os.path.join(
        #         DMP_CHECKPOINT_DIR, self.job_id + '.h5')

        #     self.keras_model = ResumableModel(
        #         self.keras_model,
        #         save_every_epochs=self.save_every_epochs,
        #         to_path=save_path)

        # fit / train model
        history = self.keras_model.fit(
            callbacks=run_callbacks,
            **prepared_config,
        )

        # Tensorflow models return a History object from their fit function,
        # but ResumableModel objects returns History.history. This smooths
        # out that incompatibility.
        if self.save_every_epochs is None or self.save_every_epochs == 0:
            history = history.history  # type: ignore

        # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
        # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
        # model.save_weights(f'./log/weights/{run_name}.h5', save_format='h5')
        # model.save(f'./log/models/{run_name}.h5', save_format='h5')

        parameters: Dict[str, Any] = parent.parameters
        parameters['output_activation'] = self.output_activation
        parameters['widths'] = widths
        parameters['num_free_parameters'] = num_free_parameters
        parameters['output_activation'] = self.output_activation
        parameters['network_structure'] = \
            jobqueue_marshal.marshal(self.network_structure)

        parameters.update(history)  # type: ignore
        parameters.update(worker.worker_info)
        
        # return the result record
        return parameters
