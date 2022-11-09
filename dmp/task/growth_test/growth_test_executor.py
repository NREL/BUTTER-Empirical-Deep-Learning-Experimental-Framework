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

from dmp.task.growth_test.growth_test_task import GrowthExperimentTask
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
class GrowthExperimentExecutor(AspectTestExecutor):
    '''
    '''

    def __call__(self, task: GrowthExperimentTask, worker, *args, **kwargs) \
            -> Dict[str, Any]:

        # check to make sure the growth scale is larger than 1
        if task.growth_scale < 1:
            raise RuntimeError('Growth scale less than one.')

        self.set_random_seeds(task)

        (
            ml_task,
            input_shape,
            output_shape,
            prepared_config,
            make_tensorflow_dataset,
        ) = self.load_and_prepare_dataset(task, val_portion=task.val_split)

        # prepare test data set
        test_data_key = 'test_data'
        test_data = (prepared_config[test_data_key][0],
                     prepared_config[test_data_key][1], test_data_key)
        del prepared_config[test_data_key]
        test_data = \
            (make_tensorflow_dataset(test_data[0], test_data[1]), test_data_key)

        # fit / train model
        history: dict = dict()
        growth_step: int = 0
        epoch_parameters: int = 0
        epochs: int = 0
        keras_model = None
        network_structure = None
        widths = None
        output_activation = None
        num_free_parameters: int = 0
        go: bool = True
        while go:

            target_size: int = int(
                math.floor(task.initial_size *
                           math.pow(task.growth_scale, growth_step)))

            # if we 'skipped' over a growth step, handle it
            if target_size <= num_free_parameters:
                growth_step += 1
                continue

            # if we topped out at the maximum size, this is the last iteration
            if target_size >= task.size:
                go = False
                target_size = task.size

            (
                network_structure,
                widths,
                num_free_parameters,
                run_loss,
                output_activation,
            ) = self.make_network(
                task,
                input_shape,
                output_shape,
                ml_task,
                target_size,
            )

            max_epochs_at_this_iteration = min(
                epochs - task.max_total_epochs,
                int(
                    math.floor((task.max_equivalent_epoch_budget * task.size) /
                               num_free_parameters)))

            if max_epochs_at_this_iteration <= 0:
                break

            # TODO: Grow Here

            # print('growing', growth_phase, num_free_parameters, ideal_size)

            # # grow each layer by the difference between old and new widths
            # config = {i: (w-old_widths[i]) for i, w in enumerate(widths)}
            # print(config)

            # # Grow
            # self.keras_model = getattr(growth_test_utils, task.growth_method)(
            #     self.keras_model,
            #     config,
            #     **self.growth_method_params,
            # )

            keras_model = self.make_keras_model(
                task,
                network_structure,
                run_loss,
            )

            prepared_config['epochs'] = max_epochs_at_this_iteration

            test_history = growth_test_utils.AdditionalValidationSets(
                [test_data],
                batch_size=task.run_config['batch_size'],
            )

            callbacks = []
            callbacks.append(test_history)

            if go:
                callbacks.append(
                    self.make_growth_trigger_callback(task.growth_trigger))

            iteration_history = self.fit_model(
                task,
                keras_model,
                prepared_config,
                callbacks,
            )

            # Add test set history into history dict.
            iteration_history.update(test_history.history)

            growth_step += 1
            num_epochs = len(iteration_history['loss'])
            epochs += num_epochs
            epoch_parameters += num_epochs * num_free_parameters

            # Add num_free_parameters to history dictionary and append to master
            # histories dictionary
            iteration_history['parameter_count'] = [num_free_parameters
                                                    ] * num_epochs

            # Add growth points to history dictionary
            iteration_history['growth_points'] = [0] * num_epochs

            # If the growth trigger is EarlyStopping and the
            # 'restore_best_weights' flag is set, indicate growth point at epoch
            # that achieves lowest val_loss else growth occured at final epoch
            if task.growth_trigger == 'EarlyStopping':
                if 'restore_best_weights' in task.growth_trigger_params and \
                        task.growth_trigger_params['restore_best_weights']:
                    iteration_history['growth_points'][numpy.argmin(
                        iteration_history['val_loss'])] = 1
                else:
                    iteration_history['growth_points'][-1] = 1

            # Extend histories dictionary
            if len(history.keys()) == 0:
                history = copy.deepcopy(iteration_history)
            else:
                for k in history.keys():
                    if type(history[k]) is list:
                        history[k].extend(iteration_history[k])

            # # Calculate config to pass to growth function
            # old_widths = widths
            # _, widths, _ = find_best_layout_for_budget_and_depth(
            #     self.inputs.shape,
            #     residual_mode,
            #     task.input_activation,
            #     task.activation,
            #     self.output_activation,
            #     task.growth_scale * ideal_size,  # Grows by the ideal size based on growth scale
            #     widths_factory(shape)(num_outputs, task.depth),
            #     layer_args,
            # )
            # # grow each layer by the difference between old and new widths
            # config = {i: (w-old_widths[i]) for i, w in enumerate(widths)}
            # print(config)

            # # Grow
            # self.keras_model = getattr(growth_test_utils, task.growth_method)(
            #     self.keras_model,
            #     config,
            #     **self.growth_method_params,
            # )

            # # Compile
            # self.keras_model.compile(
            #     loss=self.run_loss,
            #     optimizer=run_optimizer,
            #     metrics=run_metrics,
            #     run_eagerly=False,
            # )

            # if target_size >= task.size \
            #     or epoch_parameters >= task.max_equivalent_epoch_budget * task.size \
            #     or epochs >= task.max_total_epochs:
            #     break

        # Rename history keys
        history = remap_key_prefixes(history, [
            ('val_', 'validation_'),
            (test_data_key + '_', 'test_'),
            ('', 'train_'),
        ])  # type: ignore

        return self.make_result_record(
            task,
            worker,
            network_structure,  # type: ignore
            widths,  # type: ignore
            num_free_parameters,
            output_activation,  # type: ignore
            history,
        )

    def make_growth_trigger_callback(self, growth_trigger_name: str):
        if growth_trigger_name == 'EarlyStopping':
            clss = EarlyStopping
        else:
            raise NotImplementedError(
                f'Unsupported growth trigger, "{growth_trigger_name}".')
        return clss(**self.task.growth_trigger_params)  # type: ignore
