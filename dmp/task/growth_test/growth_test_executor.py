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

        self.seed = set_random_seeds(self.seed)
        self.load_dataset()
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

        # -----------
        self.make_network()
        # -----------

        # if parameters not passed, set to empty dict
        if self.growth_trigger_params == None:
            self.growth_trigger_params = {}
        if self.growth_method_params == None:
            self.growth_method_params = {}

        # check to make sure the growth scale is larger than 1
        if task.growth_scale < 1:
            raise RuntimeError('Growth scale less than one.')

        # -----------
        run_callbacks = self.make_keras_model()
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
        growth_number = math.ceil(
            math.log(task.max_size / task.size, task.growth_scale))

        # fit / train model
        histories = dict()
        epoch_parameters_left = task.max_equivalent_epoch_budget * task.max_size
        for growth_phase in range(growth_number):
            ideal_size = task.size * \
                math.pow(task.growth_scale, growth_phase)
            print('growing', growth_phase, num_free_parameters, ideal_size)

            # Calculate epoch budget left based on epoch_parameter budget of network with max size
            # epoch_budget_left is the epoch budget based on this epoch_parameter budget
            epoch_budget_left = math.ceil(
                epoch_parameters_left / ideal_size)
            print('epoch_budget_left', epoch_budget_left)
            # Calculate the total_epochs_left, which is a budget based on the max_total_epochs
            if len(histories.keys()) == 0:
                total_epochs_left = task.max_total_epochs
            else:
                total_epochs_left = task.max_total_epochs - \
                    len(histories['loss'])
            print('total_epochs_left', total_epochs_left)
            # The epochs to run the next size for becomes the minimum of these two budgets
            epochs_left = min(total_epochs_left, epoch_budget_left)
            print('per_size_epoch_budget', per_size_epoch_budget)
            # And that budget is used in the case that it is less than the per size budget set by the run_config.
            if epochs_left < per_size_epoch_budget:
                prepared_config['epochs'] = epochs_left
            else:
                prepared_config['epochs'] = per_size_epoch_budget

            # If our epoch budget has run out, stop
            if prepared_config['epochs'] == 0:
                break

            # Train
            additional_history = \
                growth_test_utils.AdditionalValidationSets(
                    [test_data],
                    batch_size=task.run_config['batch_size'],
                )

            # If we're not on our last size, use trigger
            callbacks: List[keras.callbacks.Callback] = [additional_history]

            if growth_phase < growth_number-1:
                callbacks.append(
                    self.make_growth_trigger_callback(
                        self.task.growth_trigger))

            history = self.fit_model(prepared_config, callbacks)

            # ---------- Stopped here
            print(additional_history.history.keys())

            num_epochs = len(history['loss'])

            epoch_parameters_left = epoch_parameters_left - \
                (ideal_size * num_epochs)

            # Merge history dictionaries, adding metrics from evaluation on test set.
            history.update(additional_history.history)

            # Add num_free_parameters to history dictionary and append to master
            # histories dictionary
            history['parameter_count'] = \
                [num_free_parameters for _ in range(num_epochs)]

            # Add growth points to history dictionary
            history['growth_points'] = [0 for _ in range(num_epochs)]

            # If the growth trigger is EarlyStopping and the 'restore_best_weights' flag is set,
            # indicate growth point at epoch that achieves lowest val_loss
            # else growth occured at final epoch
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

            # Calculate number of parameters in grown network
            num_free_parameters = count_trainable_parameters_in_keras_model(
                self.keras_model)

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
