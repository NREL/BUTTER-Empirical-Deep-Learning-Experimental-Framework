from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, Any, Dict, Tuple
import math
from dmp import jobqueue_interface

from jobqueue.job import Job
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayer, KerasLayerInfo
from dmp.model.network_info import NetworkInfo
from dmp.task.experiment.growth_experiment import growth_experiment_keys

from dmp.task.experiment.growth_experiment.layer_growth_info import LayerGrowthInfo
from dmp.task.experiment.growth_experiment.scaling_method.scaling_method import ScalingMethod
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import WidthScaler
from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment_keys import GrowthExperimentKeys
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.model.model_util import *
from dmp.model.model_info import ModelInfo

from dmp.task.experiment.growth_experiment.growth_experiment_keys import GrowthExperimentKeys
from dmp.task.experiment.growth_experiment.transfer_method.transfer_method import TransferMethod
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import OverlayTransfer

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.lottery_ticket_experiment.lottery_ticket_iterative_pruning_callback import LotteryTicketIterativePruningCallback
from dmp.task.experiment.training_experiment.training_experiment import TrainingExperiment
from dmp.model.model_util import find_closest_network_to_target_size_float
from dmp.worker import Worker


@dataclass
class LotteryTicketExperiment(TrainingExperiment):

    pre_prune_epochs_k: int
    num_pruning_iterations_N: int
    epochs_per_iteration: int


    end_prune_epoch: int
    
    j_steps: int
    prune_amount: float

    @property
    def version(self) -> int:
        return 0

    '''
    Masking types from LTH paper:
    + don't prune last layer
    + global and layer-wise pruning (start w/global)
    + no pruning of residual projection (conv) layers
    -> Set constraint in ModelSpec for each layer to be pruned
    '''

    def __call__(
        self,
        worker: Worker,
        job: Job,
        *args,
        **kwargs,
    ) -> ExperimentResultRecord:
        # http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf Algorithim 2

        with worker.strategy.scope():
            # tensorflow.config.optimizer.set_jit(True)
            self._set_random_seeds()
            dataset = self._load_and_prepare_dataset()
            metrics = self._autoconfigure_for_dataset(dataset)
            
            # 1: Create a network with randomly initialization W0 ∈ Rd.
            # 2: Initialize pruning mask to m = 1d.
            # TODO: import masking constraint code
            model = self._make_model(worker, self.model)
            self._compile_model(dataset, model, metrics)

            # 3: Train W0 to Wk with noise u ∼ U: Wk = A 0→k (W0, u).
            model_history = self._fit_model(
                    self.fit,
                    dataset,
                    model,
                    self._make_callbacks(),
                    epochs=self.pre_prune_epochs_k,
                )

            # 4: for n ∈ {1, . . . , N} do
            for iteration_n in range(self.num_pruning_iterations_N):
                # 5: Train m ⊙ Wk to m ⊙ WT with noise u ′∼ U:WT = Ak→T(m ⊙ Wk, u′).
                model_history = self._fit_model(
                    self.fit,
                    dataset,
                    model,
                    self._make_callbacks(),
                    epochs=self.epochs_per_iteration,
                )
                # TODO: merge histories
                # 6: Prune the lowest magnitude entries of WT that remain. Let m[i] = 0 if WT [i] is pruned.
                # TODO: prune

            # 7: Return Wk, m
            
            
            return self._make_result_record(
                worker.worker_info,
                job.id,
                dataset,
                model.network,
                history,
            )

    def growth_exp_call(self, worker: Worker, job: Job, *args,
                        **kwargs) -> ExperimentResultRecord:
        with worker.strategy.scope():
            self._set_random_seeds()
            dataset = self._load_and_prepare_dataset()
            metrics = self._autoconfigure_for_dataset(dataset)

            goal_network: NetworkInfo = self._make_network(self.model)
            # goal_network.description[self.key_names.scale_key] = 1.0
            goal_network.description[self.keys.layer_map_key] = {
                l: l
                for l in goal_network.structure.all_descendants
            }

            # from dmp.marshaling import marshal
            # print(f'goal network:')
            # pprint(
            #     marshal.marshal(
            #         goal_network.structure))

            max_total_epochs: int = self.fit['epochs']
            history: dict = {}
            model_number: int = 0
            epoch_parameters: int = 0
            epoch_count: int = 0
            src_model: Optional[ModelInfo] = None
            parent_epoch: int = 0
            on_final_iteration: bool = False
            while not on_final_iteration:

                target_size: int = int(
                    math.floor(self.initial_size *
                               math.pow(self.growth_scale, model_number)))

                # print(
                #     f'target_size {target_size}, self.initial_size {self.initial_size}, growth_step {model_number}, src_model.network.num_free_parameters {None if src_model is None else src_model.network.num_free_parameters}'
                # )
                # if we 'skipped' over a growth step, handle it
                if src_model is not None and \
                    target_size <= src_model.network.num_free_parameters:
                    model_number += 1
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
                    print(
                        f'Growing to {target_size} {network.num_free_parameters}'
                    )
                    # from dmp.marshaling import marshal
                    # pprint(
                    #     marshal.marshal(
                    #         network.structure))

                model = self._make_model_from_network(worker, network)

                max_epochs_at_this_iteration = min(
                    max_epochs_at_this_iteration,
                    math.ceil(
                        (self.max_equivalent_epoch_budget *
                         goal_network.num_free_parameters - epoch_parameters) /
                        model.network.num_free_parameters),
                )

                if max_epochs_at_this_iteration <= 0:
                    break

                fit_config = self.fit.copy()
                fit_config['epochs'] = max_epochs_at_this_iteration

                if src_model is not None:
                    self.transfer_method.transfer(
                        self._make_transfer_map(src_model, model), )

                self._compile_model(dataset, model, metrics)

                early_stopping_callback = None
                if on_final_iteration:
                    early_stopping_callback = self._make_early_stopping_callback(
                    )
                else:
                    early_stopping_callback = make_keras_instance(
                        self.growth_trigger)

                model_history = self._fit_model(
                    fit_config,
                    dataset,
                    model,
                    [early_stopping_callback],
                )

                # add additional history columns
                model_history_length = len(model_history[self.keys.train +
                                                         '_loss'])

                # model number
                model_history[self.keys.model_number] = \
                    [model_number] * model_history_length

                # model epoch
                model_history[self.keys.model_epoch] = \
                    model_history.pop(self.keys.epoch)

                # free parameter count history
                model_history[self.keys.free_parameter_count_key] = \
                    [network.num_free_parameters] * model_history_length

                last_retained_epoch = model_history_length
                if early_stopping_callback is not None \
                    and early_stopping_callback.stopped_epoch > 0:
                    last_retained_epoch -= early_stopping_callback.patience  # type: ignore

                retained = [False] * model_history_length
                for i in range(last_retained_epoch):
                    retained[i] = True
                model_history[self.keys.retained] = retained

                model_history[self.keys.epoch] = [
                    e + epoch_count
                    for e in model_history[self.keys.model_epoch]
                ]

                # Extend histories dictionary
                self._concatenate_histories(history, model_history)

                src_model = model
                model_number += 1
                epoch_count += (model_history_length - 1)
                epoch_parameters += model_history_length * model.network.num_free_parameters
                continue  # just put this here for better readability

            if src_model is None:
                raise RuntimeError(
                    f'No result record generated for task {self}.')

            src_model.network.description = goal_network.description
            return self._make_result_record(
                worker.worker_info,
                job.id,
                dataset,
                src_model.network,
                history,
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
