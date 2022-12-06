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
from dmp.task.model_data import ModelData


class GrowthExperimentExecutor(TrainingExperimentExecutor):
    '''
    '''

    def __init__(self, task: GrowthExperiment, worker):
        if task.growth_scale <= 1:
            raise RuntimeError(f'Growth scale {task.growth_scale} <= 1.')
        self.task: GrowthExperiment = task
        self.worker = worker

    def __call__(self) -> Dict[str, Any]:
        task = self.task
        self._set_random_seeds()
        dataset = self._load_and_prepare_dataset()

        # TODO get initial_size
        target_final_network = self._make_model(dataset, task.network)
        # target_final_network.si

        history: dict = {}
        growth_step: int = 0
        epoch_parameters: int = 0
        epochs: int = 0
        previous_network: Optional[ModelData] = None
        on_final_iteration: bool = False
        while not on_final_iteration:

            target_size: int = int(
                math.floor(task.initial_size *
                           math.pow(task.growth_scale, growth_step)))

            # if we 'skipped' over a growth step, handle it
            if previous_network is not None and \
                target_size <= previous_network.num_free_parameters:
                growth_step += 1
                continue

            # if we topped out at the maximum size, this is the last iteration
            if target_size >= task.size:
                on_final_iteration = True
                target_size = task.size

            model = self._make_model(dataset, self.task.size)

            max_epochs_at_this_iteration = min(
                epochs - task.max_total_epochs,
                math.floor((task.max_equivalent_epoch_budget * task.size) /
                           model.num_free_parameters))
            if max_epochs_at_this_iteration <= 0:
                break
            fit_config = deepcopy(self.task.fit_config)
            fit_config['epochs'] = max_epochs_at_this_iteration

            if previous_network is not None:
                GrowthExperimentExecutor._grow_network(
                    task.growth_method,
                    previous_network.structure,
                    previous_network.layer_to_keras_map,
                    model.structure,
                    model.layer_to_keras_map,
                )

            self._compile_model(model)
            callbacks = self._make_callbacks(on_final_iteration)
            model_history = self._fit_model(fit_config, dataset, model,
                                           callbacks)

            num_epochs = len(model_history['loss'])
            model_history['parameter_count'] = \
                [model.num_free_parameters] * num_epochs
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

            previous_network = model
            growth_step += 1
            epochs += num_epochs
            epoch_parameters += num_epochs * model.num_free_parameters
            continue  # just put this here for better readability

        if previous_network is None:
            raise RuntimeError(f'No result record generated for task {task}.')

        return self._make_result_record(previous_network, history)

    _grow_network = make_typed_config_factory(
        'growth_method',
        {
            'NetworkOverlayer': OverlayGrowthMethod,
        },
    )

    _make_growth_trigger_callback = make_typed_config_factory(
        'growth_trigger',
        {
            'EarlyStopping': keras.callbacks.EarlyStopping,
        },
    )

    def _make_callbacks(
            self, on_final_iteration: bool) -> List[keras.callbacks.Callback]:
        if on_final_iteration:
            return super()._make_callbacks()
        return [
            GrowthExperimentExecutor._make_growth_trigger_callback(
                self.task.growth_trigger)
        ]
