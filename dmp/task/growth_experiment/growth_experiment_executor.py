import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy
import tensorflow.keras as keras
from pytest import param

import dmp.task.growth_experiment.growth_experiment_utils as growth_experiment_utils
from dmp.jobqueue_interface.common import jobqueue_marshal
from dmp.task.aspect_test.aspect_test_executor import AspectTestExecutor
from dmp.task.aspect_test.aspect_test_utils import *
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.growth_experiment.network_overlayer import NetworkOverlayer
from dmp.task.task_util import remap_key_prefixes


@dataclass
class GrowthExperimentExecutor(AspectTestExecutor):
    '''
    '''

    def __call__(self, task: GrowthExperiment, worker, *args, **kwargs) \
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
        prev_network_structure = None
        prev_node_layer_map = None
        output_activation = None
        num_free_parameters: int = 0
        on_final_iteration: bool = False
        while not on_final_iteration:

            target_size: int = int(
                math.floor(task.initial_size *
                           math.pow(task.growth_scale, growth_step)))

            # if we 'skipped' over a growth step, handle it
            if target_size <= num_free_parameters:
                growth_step += 1
                continue

            # if we topped out at the maximum size, this is the last iteration
            if target_size >= task.size:
                on_final_iteration = True
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

            keras_model, node_layer_map, run_metrics, run_optimizer = \
                self.make_keras_model(task, network_structure)

            if prev_network_structure is not None and \
                prev_node_layer_map is not None:
                self.grow_network(
                    task.growth_method,
                    prev_network_structure,
                    prev_node_layer_map,
                    network_structure,
                    node_layer_map,
                )

            self.compile_keras_network(
                network_structure,
                run_loss,
                keras_model,
                run_metrics,
                run_optimizer,
            )

            prepared_config['epochs'] = max_epochs_at_this_iteration

            test_history = growth_experiment_utils.AdditionalValidationSets(
                [test_data],
                batch_size=task.run_config['batch_size'],
            )

            callbacks: List[keras.callbacks.Callback] = [test_history]

            if not on_final_iteration:
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

            num_epochs = len(iteration_history['loss'])

            # Add num_free_parameters to history dictionary and append to master
            # histories dictionary
            iteration_history['parameter_count'] = \
                [num_free_parameters] * num_epochs

            # Add growth points to history dictionary
            iteration_history['growth_points'] = \
                [0] * num_epochs

            # If the growth trigger is EarlyStopping and the
            # 'restore_best_weights' flag is set, indicate growth point at epoch
            # that achieves lowest val_loss else growth occured at final epoch
            if task.growth_trigger.get('restore_best_weights', False):
                iteration_history['growth_points'][\
                    numpy.argmin(iteration_history['val_loss'])] = 1
            else:
                iteration_history['growth_points'][-1] = 1

            # Extend histories dictionary
            if len(history.keys()) == 0:
                history = copy.deepcopy(iteration_history)
            else:
                for k in history.keys():
                    if type(history[k]) is list:
                        history[k].extend(iteration_history[k])

            prev_network_structure = network_structure
            prev_node_layer_map = node_layer_map
            growth_step += 1
            epochs += num_epochs
            epoch_parameters += num_epochs * num_free_parameters
            continue  # just put this here for better readability

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

    def grow_network(
        self,
        config: dict,
        source: NetworkModule,
        source_node_layer_map: Dict[NetworkModule, keras.layers.Layer],
        dest: NetworkModule,
        dest_node_layer_map: Dict[NetworkModule, keras.layers.Layer],
    ) -> None:
        make_from_config(
            config,
            {
                'NetworkOverlayer': NetworkOverlayer,
            },
            'growth_method',
            source,
            source_node_layer_map,
            dest,
            dest_node_layer_map,
        )

    def make_growth_trigger_callback(
        self,
        config: dict,
    ) -> keras.callbacks.Callback:
        return make_from_config(config, {
            'EarlyStopping': keras.callbacks.EarlyStopping,
        }, 'growth_trigger')
