from dataclasses import dataclass
import json
import os
import platform
import subprocess

from pytest import param
# from sympy import per

from dmp.data.pmlb import pmlb_loader
from dmp.jobqueue_interface.common import jobqueue_marshal

import tensorflow.keras.metrics as metrics
from tensorflow.keras.callbacks import EarlyStopping
from dmp.task.aspect_test.aspect_test_executor import AspectTestExecutor

from dmp.task.growth_test.growth_test_task import GrowthTestTask
from dmp.task.aspect_test.aspect_test_utils import *
import dmp.task.growth_test.growth_test_utils as growth_test_utils
from dmp.task.task import Parameter
from typing import Optional, Dict, Any

import pandas
import numpy
import sys
import copy
import math

from dmp.task.task_util import remap_key_prefixes

# from keras_buoy.models import ResumableModel


@dataclass
class GrowthTestExecutor(AspectTestExecutor):
    '''
    '''

    def __call__(self, parent: GrowthTestTask, worker, *args, **kwargs) \
            -> Dict[str, Any]:

        # for easy access
        task: GrowthTestTask = self.task  # type: ignore

        self.set_random_seeds()
        self.dataset_series, self.inputs, self.outputs = self.load_dataset()

        # -----------

        # prepare dataset shuffle, split, and label noise:
        prepared_config = prepare_dataset(
            task.test_split_method,
            task.test_split,
            task.label_noise,
            task.run_config,
            str(self.dataset_series['Task']),
            self.inputs,
            self.outputs,
            val_portion=task.val_split,
        )
        test_data = (prepared_config['test_data'][0],
                     prepared_config['test_data'][1], 'test_data')
        del prepared_config['test_data']

        # Get the per size epoch budget
        per_size_epoch_budget = prepared_config['epochs']

        # Use one of these budgets during growth. If all are none, raise error, else set them to infinity.
        if self.max_total_epochs == None and self.max_equivalent_epoch_budget == None and per_size_epoch_budget == None:
            raise ValueError(
                "Must specify value for max_total_epochs, max_equivalent_epoch_budget, or run_config['epochs']")
        if self.max_total_epochs == None:
            self.max_total_epochs = numpy.inf
        if self.max_equivalent_epoch_budget == None:
            self.max_equivalent_epoch_budget = numpy.inf
        if per_size_epoch_budget == None:
            per_size_epoch_budget = numpy.inf

        print(per_size_epoch_budget)

        # if parameters not passed, set to empty dict
        if self.growth_trigger_params == None:
            self.growth_trigger_params = {}
        if self.growth_method_params == None:
            self.growth_method_params = {}

        # check to make sure the growth scale is larger than 1
        if task.growth_scale < 1:
            raise RuntimeError('Growth scale less than one.')

        # -----------

        make_tensorflow_dataset = \
            self.make_tensorflow_dataset_converter(prepared_config)
        self.setup_tensorflow_training_and_validation_datasets(
            make_tensorflow_dataset,
            prepared_config,
        )
        # -----------

        test_data = \
            (make_tensorflow_dataset(test_data[0], test_data[1]), 'test_data')

        # Calculate number of growth phases
        num_growth_steps = math.ceil(
            math.log(task.max_size / task.size, task.growth_scale))

        # fit / train model
        histories: dict = dict()
        growth_step: int = 0
        max_size_threshold = max(1.0-.1, 1.0/(task.growth_scale)) * task.size
        epoch_parameters: int = 0
        epochs: int = 0
        go: bool = True
        while go:

            target_size: int = min(
                task.size,
                int(math.floor(
                    task.initial_size *
                    math.pow(task.growth_scale, growth_step))))

            # When we reach or get very close to the max size, we just go to it
            if target_size > max_size_threshold:
                target_size = task.size
                go = False

            # print('growing', growth_phase, num_free_parameters, ideal_size)

            parameter_count = self.make_network(target_size)

            
            # TODO: Grow Here


            self.make_keras_model()  # callbacks discarded (replaced later)

            max_epochs_at_this_iteration = min(
                epochs - task.max_total_epochs,
                int(math.floor(
                    (task.max_equivalent_epoch_budget * task.size) /
                    parameter_count))
            )

            if max_epochs_at_this_iteration <= 0:
                break

            prepared_config['epochs'] = max_epochs_at_this_iteration

            # Train
            
            additional_history = \
                growth_test_utils.AdditionalValidationSets(
                    [test_data],
                    batch_size=task.run_config['batch_size'],
                )

            callbacks = []
            callbacks.append(additional_history)

            if go:
                callbacks.append(
                    self.make_growth_trigger_callback(
                        task.growth_trigger))

            history = self.fit_model(prepared_config, callbacks)

            # Add test set history into history dict.
            history.update(additional_history.history)

            num_epochs = len(history['loss'])
            epochs += num_epochs
            epoch_parameters += num_epochs * parameter_count

            # Add num_trainable_parameters to history dictionary and append to master
            # histories dictionary
            history['parameter_count'] = [parameter_count] * num_epochs

            # Add growth points to history dictionary
            history['growth_points'] = [0] * num_epochs

            # If the growth trigger is EarlyStopping and the 
            # 'restore_best_weights' flag is set, indicate growth point at epoch
            # that achieves lowest val_loss else growth occured at final epoch
            if task.growth_trigger == 'EarlyStopping':
                if 'restore_best_weights' in self.growth_trigger_params and \
                        self.growth_trigger_params['restore_best_weights']:
                    history['growth_points'][
                        numpy.argmin(history['val_loss'])] = 1
                else:
                    history['growth_points'][-1] = 1

            # Extend histories dictionary
            if len(histories.keys()) == 0:
                histories = copy.deepcopy(history)
            else:
                for k in histories.keys():
                    if type(histories[k]) is list:
                        histories[k].extend(history[k])

            # Calculate config to pass to growth function
            old_widths = widths
            _, widths, _ = find_best_layout_for_budget_and_depth(
                self.inputs.shape,
                residual_mode,
                task.input_activation,
                task.activation,
                self.output_activation,
                task.growth_scale * ideal_size,  # Grows by the ideal size based on growth scale
                widths_factory(shape)(num_outputs, task.depth),
                layer_args,
            )
            # grow each layer by the difference between old and new widths
            config = {i: (w-old_widths[i]) for i, w in enumerate(widths)}
            print(config)

            # Grow
            self.keras_model = \
                getattr(growth_test_utils, task.growth_method)(
                    self.keras_model,
                    config,
                    **self.growth_method_params,
                )

            # Compile
            self.keras_model.compile(
                loss=self.run_loss,
                optimizer=run_optimizer,
                metrics=run_metrics,
                run_eagerly=False,
            )

            # if target_size >= task.size \
            #     or epoch_parameters >= task.max_equivalent_epoch_budget * task.size \
            #     or epochs >= task.max_total_epochs:
            #     break

        # Rename history keys
        histories = remap_key_prefixes(
            histories, [
                ('val_', 'validation_'),
                ('test_data_', 'test_'),
                ('', 'train_'),
            ])  # type: ignore

        parameters: Dict[str, Any] = parent.parameters
        parameters['output_activation'] = self.output_activation
        parameters['widths'] = self.widths
        parameters['num_free_parameters'] = num_free_parameters
        parameters['output_activation'] = self.output_activation
        parameters['network_structure'] = \
            jobqueue_marshal.marshal(self.network_structure)

        parameters.update(histories)

        # return the result record
        return parameters

    def make_growth_trigger_callback(self, growth_trigger_name: str):
        if growth_trigger_name == 'EarlyStopping':
            clss = EarlyStopping
        else:
            raise NotImplementedError(
                f'Unsupported growth trigger, "{growth_trigger_name}".')
        return clss(**self.growth_trigger_params)
