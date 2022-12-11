import copy
import math
from typing import Any, Dict, Optional
import tensorflow.keras as keras
import numpy
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayer, KerasLayerInfo

import dmp.task.growth_experiment.growth_experiment_utils as growth_experiment_utils
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.growth_experiment.growth_method.overlay_growth import OverlayGrowth
from dmp.task.growth_experiment.layer_growth_info import LayerGrowthInfo
from dmp.task.growth_experiment.scaling_method.width_scaler import WidthScaler
from dmp.layer.visitor.keras_interface.keras_utils import make_typed_keras_config_factory
from dmp.task.task_util import *
from dmp.task.training_experiment.training_experiment_executor import TrainingExperimentExecutor
from dmp.model.model_info import ModelInfo

layer_map_key: str = 'layer_map'
scale_key: str = 'scale'
growth_points_key: str = 'growth_points'


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

        final_network: NetworkInfo = self._make_network(self.task.model)
        history: dict = {}
        growth_step: int = 0
        epoch_parameters: int = 0
        epochs: int = 0
        src_model: Optional[ModelInfo] = None
        on_final_iteration: bool = False
        while not on_final_iteration:

            target_size: int = int(
                math.floor(task.initial_size *
                           math.pow(task.growth_scale, growth_step)))

            # if we 'skipped' over a growth step, handle it
            if src_model is not None and \
                target_size <= src_model.network.num_free_parameters:
                growth_step += 1
                continue

            # if we topped out at the maximum size, this is the last iteration
            network = None
            if target_size >= final_network.num_free_parameters:
                on_final_iteration = True
                target_size = final_network.num_free_parameters
                network = final_network
            else:

                def make_network(scale: float) -> NetworkInfo:
                    scaled, layer_map = WidthScaler(final_network.structure,
                                                    scale)()
                    description = final_network.description.copy()
                    description[scale_key] = scale
                    description[layer_map_key] = layer_map
                    return NetworkInfo(scaled, description)

                delta, network = find_closest_network_to_target_size_float(
                    target_size,
                    make_network,
                )

            model = self._make_model_from_network(network)

            max_epochs_at_this_iteration = min(
                epochs - task.max_total_epochs,
                math.floor((task.max_equivalent_epoch_budget *
                            final_network.num_free_parameters) /
                           model.network.num_free_parameters))
            if max_epochs_at_this_iteration <= 0:
                break
            fit_config = copy.deepcopy(self.task.fit_config)
            fit_config['epochs'] = max_epochs_at_this_iteration

            if src_model is not None:
                self._grow_network(
                    task.growth_method,
                    src_model,
                    model,
                    self._make_growth_map(src_model, model),
                )

            self._compile_model(dataset, model)
            callbacks = self._make_callbacks(on_final_iteration)
            model_history = self._fit_model(
                fit_config,
                dataset,
                model,
                callbacks,
            )

            num_epochs = len(model_history['loss'])
            model_history[scale_key] = [network.description[scale_key]
                                        ] * num_epochs
            model_history['num_free_parameters_history'] = \
                [network.num_free_parameters] * num_epochs
            model_history[growth_points_key] = [0] * num_epochs

            # If the growth trigger is EarlyStopping and the
            # 'restore_best_weights' flag is set, indicate growth point at epoch
            # that achieves lowest val_loss else growth occured at final epoch
            if task.growth_trigger.get('restore_best_weights', False):
                model_history[growth_points_key][numpy.argmin(
                    model_history['val_loss'])] = 1
            else:
                model_history[growth_points_key][-1] = 1

            # Extend histories dictionary
            if len(history.keys()) == 0:
                history = copy.deepcopy(model_history)
            else:
                for k, v in history.items():
                    if type(v) is list:
                        v.extend(model_history[k])

            src_model = model
            growth_step += 1
            epochs += num_epochs
            epoch_parameters += num_epochs * model.network.num_free_parameters
            continue  # just put this here for better readability

        if src_model is None:
            raise RuntimeError(f'No result record generated for task {task}.')

        src_model.network.description = final_network.description
        return self._make_result_record(dataset, src_model, history)

    _grow_network = make_typed_keras_config_factory(
        'growth_method',
        {
            'NetworkOverlayer': OverlayGrowth,
        },
    )

    _make_growth_trigger_callback = make_typed_keras_config_factory(
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
            model.network.description[layer_map_key],
            model.keras_network.layer_to_keras_map,
        )
