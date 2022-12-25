import copy
import math
import pprint
from typing import Any, Dict, Optional
import tensorflow.keras as keras
import numpy
from dmp import jobqueue_interface
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayer, KerasLayerInfo

import dmp.task.growth_experiment.growth_experiment_utils as growth_experiment_utils
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.growth_experiment.growth_method.overlay_growth_method import OverlayGrowthMethod
from dmp.task.growth_experiment.growth_trigger.proportional_stopping import ProportionalStopping
from dmp.task.growth_experiment.layer_growth_info import LayerGrowthInfo
from dmp.task.growth_experiment.scaling_method.width_scaler import WidthScaler
from dmp.layer.visitor.keras_interface.keras_utils import make_keras_instance, register_custom_keras_types
from dmp.task.growth_experiment.training_experiment_keys import GrowthExperimentKeys
from dmp.task.task_result_record import TaskResultRecord
from dmp.task.task_util import *
from dmp.task.training_experiment.training_experiment_executor import TrainingExperimentExecutor
from dmp.model.model_info import ModelInfo


register_custom_keras_types({
    'NetworkOverlayer': OverlayGrowthMethod,
    'ProportionalStopping': ProportionalStopping,
})


class GrowthExperimentExecutor(TrainingExperimentExecutor):
    '''
    '''
    key_names = GrowthExperimentKeys()

    def __init__(self, task: GrowthExperiment, worker):
        super().__init__(task, worker)
        if task.growth_scale <= 1:
            raise RuntimeError(f'Growth scale {task.growth_scale} <= 1.')

    def __call__(self) -> TaskResultRecord:
        task = self.task
        self._set_random_seeds()
        dataset = self._load_and_prepare_dataset()
        metrics = self._autoconfigure_for_dataset(dataset)

        goal_network: NetworkInfo = self._make_network(task.model)
        goal_network.description[self.key_names.scale_key] = 1.0
        goal_network.description[self.key_names.layer_map_key] = {
            l: l
            for l in goal_network.structure.all_descendants
        }

        max_total_epochs: int = task.fit['epochs']
        history: dict = {}
        growth_step: int = 0
        epoch_parameters: int = 0
        epoch_count: int = 0
        src_model: Optional[ModelInfo] = None
        parent_epoch: int = 0
        on_final_iteration: bool = False
        while not on_final_iteration:

            target_size: int = int(
                math.floor(task.initial_size *
                           math.pow(task.growth_scale, growth_step)))

            print(
                f'target_size {target_size}, task.initial_size {task.initial_size}, growth_step {growth_step}, src_model.network.num_free_parameters {None if src_model is None else src_model.network.num_free_parameters}'
            )
            # if we 'skipped' over a growth step, handle it
            if src_model is not None and \
                target_size <= src_model.network.num_free_parameters:
                growth_step += 1
                continue

            max_epochs_at_this_iteration = max_total_epochs - epoch_count

            # if we topped out at the maximum size, this is the last iteration
            network = None
            if target_size >= goal_network.num_free_parameters:
                on_final_iteration = True
                target_size = goal_network.num_free_parameters
                network = goal_network
            else:
                max_epochs_at_this_iteration = min(
                    max_epochs_at_this_iteration,
                    task.max_epochs_per_stage,
                )

                def make_network(scale: float) -> NetworkInfo:
                    scaled, layer_map = WidthScaler(goal_network.structure,
                                                    scale)()
                    description = goal_network.description.copy()
                    description[self.key_names.scale_key] = scale
                    description[self.key_names.layer_map_key] = layer_map
                    return NetworkInfo(scaled, description)

                delta, network = find_closest_network_to_target_size_float(
                    target_size,
                    make_network,
                )
                print(
                    f'Growing to {target_size} {network.num_free_parameters}')
                # pprint.pprint(
                #     jobqueue_interface.jobqueue_marshal.marshal(
                #         network.description))

            model = self._make_model_from_network(network)

            max_epochs_at_this_iteration = min(
                max_epochs_at_this_iteration,
                math.ceil(
                    (task.max_equivalent_epoch_budget *
                     goal_network.num_free_parameters - epoch_parameters) /
                    model.network.num_free_parameters),
            )

            if max_epochs_at_this_iteration <= 0:
                break

            fit_config = task.fit.copy()
            fit_config['epochs'] = max_epochs_at_this_iteration

            if src_model is not None:
                task.growth_method.grow(
                    src_model,
                    model,
                    self._make_growth_map(src_model, model),
                )

            self._compile_model(dataset, model, metrics)

            early_stopping_callback = None
            if on_final_iteration:
                early_stopping_callback = self._make_early_stopping_callback()
            else:
                early_stopping_callback = make_keras_instance(
                    task.growth_trigger)

            model_history = self._fit_model(
                fit_config,
                dataset,
                model,
                [early_stopping_callback],
            )

            # add additional history columns
            model_history_length = len(model_history[self.key_names.train_key + '_loss'])

            # # model scale as % of target model's num parameters
            # model_history[scale_key] = \
            #     [network.description[scale_key]] * num_epochs

            '''
            + aggregate statistics:
                + growth points (medians, iqr)
                + parameter count history
                + losses for non-discarded/rewound epochs 
                + "overshoot" burden? cumulative # of discarded epochs?
                + epoch 
                + start training
                + stop training

                
            + new run statistics:
                + start timestamp
                + end timestamp
                + per epoch
                    + start training/epoch
                    + end train (or delta?)
                    + 
                + total time:
                    + training
                    + validating
                    + testing
            '''

            # free parameter count history
            model_history[self.key_names.free_parameter_count_key] = \
                [network.num_free_parameters] * model_history_length

            # epoch of source model
            parent_epochs: List[Optional[int]] = [None] * model_history_length
            if src_model is not None:
                parent_epochs[0] = parent_epoch
            model_history[self.key_names.parent_epoch_key] = parent_epoch

            parent_epoch = model_history_length - 1
            if early_stopping_callback is not None \
                and early_stopping_callback.stopped_epoch > 0:
                parent_epoch -= early_stopping_callback.patience  # type: ignore

            growth_source = [False] * model_history_length
            growth_source[parent_epoch] = True
            model_history[self.key_names.growth_source_key] = growth_source

            # Extend histories dictionary
            self._append_history_dicts(history, model_history)

            parent_epoch = history[self.key_names.epoch_key][-1]
            if early_stopping_callback is not None \
                and early_stopping_callback.stopped_epoch > 0:
                parent_epoch -= early_stopping_callback.patience  # type: ignore

            src_model = model
            growth_step += 1
            epoch_count += (model_history_length - 1)
            epoch_parameters += model_history_length * model.network.num_free_parameters
            continue  # just put this here for better readability

        if src_model is None:
            raise RuntimeError(f'No result record generated for task {task}.')

        src_model.network.description = goal_network.description
        return self._make_result_record(dataset, src_model, history)

    def _make_callbacks(
        self,
        on_final_iteration: bool = True,
    ) -> List[keras.callbacks.Callback]:
        if on_final_iteration:
            return super()._make_callbacks()
        result = []
        growth_trigger = self.task.growth_trigger
        if growth_trigger is not None:
            result.append(make_keras_instance(growth_trigger))
            pprint.pprint(growth_trigger)
        return result

    def _make_growth_map(self, src: ModelInfo, dest: ModelInfo):
        src_layer_map, src_layer_to_keras_map = self._get_layer_maps(src)
        dest_layer_map, dest_layer_to_keras_map = self._get_layer_maps(dest)
        return {
            src_layer: LayerGrowthInfo(
                src_layer_to_keras_map[src_layer],
                dest_layer_to_keras_map[dest_layer_map[ref_layer]],
            )
            for ref_layer, src_layer in src_layer_map.items()
            if ref_layer in dest_layer_map
        }

    def _get_layer_maps(
        self,
        model: ModelInfo,
    ) -> Tuple[Dict[Layer, Layer], Dict[Layer, KerasLayerInfo]]:
        return (
            model.network.description[self.key_names.layer_map_key],
            model.keras_network.layer_to_keras_map,
        )
