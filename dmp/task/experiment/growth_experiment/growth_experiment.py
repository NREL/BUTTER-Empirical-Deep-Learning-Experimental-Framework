from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, Any, Dict, Tuple
import math

from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayer, KerasLayerInfo
from dmp.model.network_info import NetworkInfo
from dmp.task.experiment.growth_experiment import growth_experiment_keys

from dmp.task.experiment.growth_experiment.layer_growth_info import LayerGrowthInfo
from dmp.task.experiment.growth_experiment.scaling_method.scaling_method import (
    ScalingMethod,
)
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)
from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment_keys import (
    GrowthExperimentKeys,
)
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.model.model_util import *
from dmp.model.model_info import ModelInfo

from dmp.task.experiment.growth_experiment.growth_experiment_keys import (
    GrowthExperimentKeys,
)
from dmp.task.experiment.growth_experiment.transfer_method.transfer_method import (
    TransferMethod,
)
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.model.model_util import find_closest_network_to_target_size_float
from dmp.worker_task_context import WorkerTaskContext


@dataclass
class GrowthExperiment(TrainingExperiment):
    growth_trigger: dict = field(
        default_factory=lambda: make_keras_kwcfg(
            'EarlyStopping',
            restore_best_weights=True,
            monitor='val_loss',
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto',
            baseline=None,
            # start_from_epoch=0,
        )
    )

    scaling_method: ScalingMethod = field(default_factory=WidthScaler)
    transfer_method: TransferMethod = field(default_factory=OverlayTransfer)
    growth_scale: float = 2.0
    # num_scales: int = 1024
    initial_size: int = 1024
    max_epochs_per_stage: int = 3000
    max_equivalent_epoch_budget: int = 3000

    keys = growth_experiment_keys.keys

    @property
    def version(self) -> int:
        return super().version + 2

    def __call__(
        self, 
        context: WorkerTaskContext, 
    ) -> ExperimentResultRecord:
        self._set_random_seeds()
        dataset, metrics = self._load_and_prepare_dataset()

        goal_network: NetworkInfo = self._make_network(self.model)
        # goal_network.description[self.key_names.scale_key] = 1.0
        goal_network.description[self.keys.layer_map_key] = {
            l: l for l in goal_network.structure.layers
        }

        # from dmp.marshaling import marshal
        # print(f'goal network:')
        # pprint(
        #     marshal.marshal(
        #         goal_network.structure))

        max_total_epochs: int = self.fit['epochs']
        experiment_history: Dict[str, Any] = {}
        model_number: int = 0
        epoch_parameters: int = 0
        src_model: Optional[ModelInfo] = None
        on_final_iteration: bool = False
        while not on_final_iteration:
            target_size: int = int(
                math.floor(
                    self.initial_size * math.pow(self.growth_scale, model_number)
                )
            )

            # print(
            #     f'target_size {target_size}, self.initial_size {self.initial_size}, growth_step {model_number}, src_model.network.num_free_parameters {None if src_model is None else src_model.network.num_free_parameters}'
            # )

            max_epochs_at_this_iteration = max_total_epochs - self._get_last_epoch(
                experiment_history
            )

            # if we topped out at the maximum size, this is the last iteration
            network = None
            if target_size >= goal_network.num_free_parameters:
                on_final_iteration = True
                target_size = goal_network.num_free_parameters
                network = goal_network
            else:
                max_epochs_at_this_iteration = min(
                    max_epochs_at_this_iteration,
                    self.max_epochs_per_stage,
                )

                def make_network(scale: float) -> NetworkInfo:
                    scaled, layer_map = self.scaling_method.scale(
                        goal_network.structure,
                        scale,
                    )
                    description = goal_network.description.copy()
                    # description[self.key_names.scale_key] = scale
                    description[self.keys.layer_map_key] = layer_map
                    return NetworkInfo(scaled, description)

                delta, network = find_closest_network_to_target_size_float(
                    target_size,
                    make_network,
                )
                # from dmp.marshaling import marshal
                # pprint(
                #     marshal.marshal(
                #         network.structure))

                if (
                    src_model is not None
                    and network.num_free_parameters
                    <= src_model.network.num_free_parameters
                ):
                    model_number += 1
                    continue

                print(f'Growing to {target_size} {network.num_free_parameters}')

            max_epochs_at_this_iteration = min(
                max_epochs_at_this_iteration,
                math.ceil(
                    (
                        self.max_equivalent_epoch_budget
                        * goal_network.num_free_parameters
                        - epoch_parameters
                    )
                    / network.num_free_parameters
                ),
            )

            if max_epochs_at_this_iteration <= 0:
                break

            model = self._make_model_from_network(network, metrics)
            if src_model is None:
                pass
            else:
                self.transfer_method.transfer(
                    self._make_transfer_map(src_model, model),
                )

            early_stopping = None
            if on_final_iteration:
                early_stopping = self._make_early_stopping_callback()
            else:
                early_stopping = make_keras_instance(self.growth_trigger)

            self._fit_model(
                context,
                self.fit,
                dataset,
                model,
                [early_stopping],
                epochs=max_epochs_at_this_iteration,
                experiment_history=experiment_history,
            )

            src_model = model
            model_number += 1
            epoch_parameters += (
                self._get_last_epoch(experiment_history)
                * model.network.num_free_parameters
            )
            continue  # just put this here for better readability

        if src_model is None:
            raise RuntimeError(f'No result record generated for task {self}.')

        src_model.network.description = goal_network.description
        return self._make_result_record(
            context,
            dataset,
            src_model.network,
            experiment_history,
        )

    def _make_transfer_map(
        self,
        src: ModelInfo,
        dst: ModelInfo,
    ) -> List[LayerGrowthInfo]:
        src_layer_map, src_layer_to_keras_map = self._get_layer_maps(src)
        dst_layer_map, dst_layer_to_keras_map = self._get_layer_maps(dst)

        layer_growth_mapping = []
        for ref_layer, src_layer in src_layer_map.items():
            src_info = src_layer_to_keras_map[src_layer]
            dst_info = None
            dst_layer = dst_layer_map.get(ref_layer, None)
            if dst_layer is not None:
                dst_info = dst_layer_to_keras_map[dst_layer]
            layer_growth_mapping.append(LayerGrowthInfo(src_info, dst_info))

        return layer_growth_mapping

    def _get_layer_maps(
        self,
        model: ModelInfo,
    ) -> Tuple[Dict[Layer, Layer], Dict[Layer, KerasLayerInfo]]:
        return (
            model.network.description[self.keys.layer_map_key],
            model.keras_network.layer_to_keras_map,
        )
        '''
        + aggregate statistics:
            + median, iqr growth points 
                -> model number
            + median, iqr parameter count -> parameter count
            + median, iqr losses for retained epochs
                -> 'discarded' bool
                -> 'source model', 'source epoch'?
            + "overshoot" burden? 
                + cumulative # of discarded epochs?
        '''
