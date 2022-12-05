import copy
import math
from typing import Any, Dict, Optional

import numpy

import dmp.task.growth_experiment.growth_experiment_utils as growth_experiment_utils
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.growth_experiment.growth_methods.overlay_growth_method import OverlayGrowthMethod
from dmp.task.task_util import remap_key_prefixes
from dmp.task.training_experiment.training_experiment_utils import *
from dmp.task.training_experiment.training_experiment_executor import TrainingExperimentExecutor
from dmp.task.training_experiment.network import Network


class GrowthExperimentExecutor(TrainingExperimentExecutor):
    '''
    '''

    def __call__(self) -> Dict[str, Any]:
        return self.result

    def __init__(self, task: GrowthExperiment, worker, *args, **kwargs):

        # check to make sure the growth scale is larger than 1
        if task.growth_scale < 1:
            raise RuntimeError('Growth scale less than one.')

        self.set_random_seeds(task)

        (
            ml_task,
            input_shape,
            output_shape,
            fit_config,
            test_data,
        ) = self.load_and_prepare_dataset(task)

        history: dict = {}
        growth_step: int = 0
        epoch_parameters: int = 0
        epochs: int = 0
        previous_network: Optional[Network] = None
        on_final_iteration: bool = False
        while not on_final_iteration:

            target_size: int = int(
                math.floor(task.initial_size *
                           math.pow(task.growth_scale, growth_step)))

            # if we 'skipped' over a growth step, handle it
            if previous_network is not None and target_size <= previous_network.num_free_parameters:
                growth_step += 1
                continue

            # if we topped out at the maximum size, this is the last iteration
            if target_size >= task.size:
                on_final_iteration = True
                target_size = task.size

            network: Network = self.make_network(
                task,
                input_shape,
                output_shape,
                target_size,
                ml_task,
            )

            max_epochs_at_this_iteration = min(
                epochs - task.max_total_epochs,
                math.floor((task.max_equivalent_epoch_budget * task.size) /
                           network.num_free_parameters))
            fit_config['epochs'] = max_epochs_at_this_iteration
            if max_epochs_at_this_iteration <= 0:
                break

            if previous_network is not None:
                self.grow_network(task, previous_network, network)

            network.compile_model(task.optimizer)

            callbacks = self.make_callbacks(task, test_data)
            if on_final_iteration:
                self.add_early_stopping_callback(task, callbacks)
            else:
                callbacks.append(
                    self.make_growth_trigger_callback(task.growth_trigger))

            model_history = self.fit_model(fit_config, network, callbacks)

            num_epochs = len(model_history['loss'])
            model_history['parameter_count'] = \
                [network.num_free_parameters] * num_epochs
            model_history['growth_points'] = [0] * num_epochs

            # If the growth trigger is EarlyStopping and the
            # 'restore_best_weights' flag is set, indicate growth point at epoch
            # that achieves lowest val_loss else growth occured at final epoch
            if task.growth_trigger.get('restore_best_weights', False):
                model_history['growth_points'][numpy.argmin(
                    model_history['val_loss'])] = 1
            else:
                model_history['growth_points'][-1] = 1

            # Extend histories dictionary
            if len(history.keys()) == 0:
                history = copy.deepcopy(model_history)
            else:
                for k, v in history.items():
                    if type(v) is list:
                        v.extend(model_history[k])

            previous_network = network
            growth_step += 1
            epochs += num_epochs
            epoch_parameters += num_epochs * network.num_free_parameters
            continue  # just put this here for better readability

        if previous_network is None:
            raise RuntimeError(f'No result record generated for task {task}.')

        self.result = self.make_result_record(
            task,
            worker,
            previous_network,
            history,
        )

    def grow_network(
        self,
        task: GrowthExperiment,
        source: Network,
        dest: Network,
    ) -> None:
        make_from_typed_config(
            task.growth_method,
            {
                'NetworkOverlayer': OverlayGrowthMethod,
            },
            'growth_method',
            source.network_structure,
            source.layer_to_keras_map,
            dest.network_structure,
            dest.layer_to_keras_map,
        )

    def make_growth_trigger_callback(
        self,
        config: dict,
    ) -> keras.callbacks.Callback:
        return make_from_typed_config(
            config, {
                'EarlyStopping': keras.callbacks.EarlyStopping,
            }, 'growth_trigger')
