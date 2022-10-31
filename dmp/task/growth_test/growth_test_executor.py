from dataclasses import dataclass
import json
import os
import platform
import subprocess

from pytest import param

from dmp.data.pmlb import pmlb_loader
from dmp.jobqueue_interface.common import jobqueue_marshal

import tensorflow.keras.metrics as metrics
from tensorflow.keras.callbacks import EarlyStopping

from dmp.task.growth_test.growth_test_task import GrowthTestTask
from dmp.task.aspect_test.aspect_test_utils import *
import dmp.task.growth_test.growth_test_utils as growth_test_utils
from dmp.task.task import Parameter
from typing import Optional, Dict, Any

import pandas
import numpy
import sys
import copy

# from keras_buoy.models import ResumableModel

_datasets = pmlb_loader.load_dataset_index()


@dataclass
class GrowthTestExecutor(GrowthTestTask):
    '''
    '''

    output_activation: Optional[str] = None
    tensorflow_strategy: Optional[tensorflow.distribute.Strategy] = None
    keras_model: Optional[tensorflow.keras.Model] = None
    run_loss: Optional[tensorflow.keras.losses.Loss] = None
    network_structure: Optional[NetworkModule] = None
    dataset_series: Optional[pandas.Series] = None
    inputs: Optional[numpy.ndarray] = None
    outputs: Optional[numpy.ndarray] = None

    def __call__(self, parent: GrowthTestTask, worker, *args, **kwargs) \
            -> Dict[str, Any]:
        # Configure hardware
        if self.tensorflow_strategy is None:
            self.tensorflow_strategy = tensorflow.distribute.get_strategy()

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
            self.dataset_series['Task'],
            self.inputs,
            self.outputs,
            val_portion=self.val_split,
        )
        test_data = (prepared_config['test_data'][0],prepared_config['test_data'][1],'test_data')
        del prepared_config['test_data']

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
            raise ValueError(f'Could not find conformant network error : {relative_error}%, delta : {delta}, size: {self.size}.')

        # if parameters not passed, set to empty dict
        if self.growth_trigger_params == None:
            self.growth_trigger_params = {}
        if self.growth_method_params == None:
            self.growth_method_params = {}

        # check to make sure the growth scale is larger than 1
        assert self.growth_scale >= 1, 'Growth scale less than one.'

        # Create and execute network using Keras
        with self.tensorflow_strategy.scope():
            # Build Keras model
            self.keras_model = make_keras_network_from_network_module(
                self.network_structure)
            if len(self.keras_model.inputs) != 1:
                print(
                    f'weird error: {len(self.keras_model.inputs)}, {json.dumps(jobqueue_marshal.marshal(self.network_structure))}')
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
            )

            num_free_parameters = count_trainable_parameters_in_keras_model(
                self.keras_model)
            assert count_num_free_parameters(self.network_structure) == num_free_parameters, \
                'Wrong number of trainable parameters'

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
            histories = dict()
            while num_free_parameters < self.max_size:
                print(num_free_parameters)
                # Train
                additional_history = growth_test_utils.AdditionalValidationSets([test_data],
                                    batch_size=self.run_config['batch_size'])
                history = self.keras_model.fit(
                    callbacks=[getattr(sys.modules[__name__], self.growth_trigger)(**self.growth_trigger_params),
                               additional_history],
                    **prepared_config,
                )
                print(additional_history.history.keys())

                # Tensorflow models return a History object from their fit function,
                # but ResumableModel objects returns History.history. This smooths
                # out that incompatibility.
                if self.save_every_epochs is None or self.save_every_epochs == 0:
                    history = history.history
                
                # Merge history dictionaries, adding metrics from evaluation on test set.
                history = {**additional_history.history, **history}

                # Add num_free_parameters to history dictionary and append to master histories dictionary
                history['parameter_count'] = [num_free_parameters for _ in range(len(history['loss']))]

                # Add growth points to history dictionary
                history['growth_points'] = [0 for _ in range(len(history['loss']))]

                # If the growth trigger is EarlyStopping and the 'restore_best_weights' flag is set,
                # indicate growth point at epoch that achieves lowest val_loss
                # else growth occured at final epoch
                if self.growth_trigger == 'EarlyStopping':
                    if 'restore_best_weights' in self.growth_trigger_params and self.growth_trigger_params['restore_best_weights']:
                        history['growth_points'][numpy.argmin(history['val_loss'])] = 1
                    else:
                        history['growth_points'][-1] = 1

                # Extend histories dictionary
                if len(histories.keys())==0:
                    histories = copy.deepcopy(history)
                else:
                    for k in histories.keys():
                        if type(histories[k]) is list:
                            histories[k].extend(history[k])

                # Calculate config to pass to growth function
                old_widths = widths
                _, widths, _ = find_best_layout_for_budget_and_depth(
                    self.inputs,
                    residual_mode,
                    self.input_activation,
                    self.activation,
                    self.output_activation,
                    self.growth_scale * num_free_parameters,
                    widths_factory(shape)(num_outputs, self.depth),
                    self.depth,
                    shape,
                    layer_args,
                )
                config = {i:(w-old_widths[i]) for i,w in enumerate(widths)} # grow each layer by the difference between old and new widths
                print(config)

                # Grow
                self.keras_model = getattr(growth_test_utils, self.growth_method)(self.keras_model,config,**self.growth_method_params)
            
                # Compile
                self.keras_model.compile(
                    # loss='binary_crossentropy', # binary classification
                    # loss='categorical_crossentropy', # categorical classification (one hot)
                    loss=self.run_loss,  # regression
                    optimizer=run_optimizer,
                    # optimizer='rmsprop',
                    # metrics=['accuracy'],
                    metrics=run_metrics,
                )

                # Calculate number of parameters in grown network
                num_free_parameters = count_trainable_parameters_in_keras_model(self.keras_model)

            # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
            # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
            # model.save_weights(f'./log/weights/{run_name}.h5', save_format='h5')
            # model.save(f'./log/models/{run_name}.h5', save_format='h5')

            parameters: Dict[str, any] = parent.parameters
            parameters['output_activation'] = self.output_activation
            parameters['widths'] = widths
            parameters['num_free_parameters'] = num_free_parameters
            parameters['output_activation'] = self.output_activation
            parameters['network_structure'] = \
                jobqueue_marshal.marshal(self.network_structure)

            parameters.update(histories)

            # return the result record
            return parameters
