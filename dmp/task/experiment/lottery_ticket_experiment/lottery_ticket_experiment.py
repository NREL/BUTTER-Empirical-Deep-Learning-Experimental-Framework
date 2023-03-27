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

    questions:
        + Is the LT determined only by itself, or does it rely on the larger network?
            + Can we recover an LT when transplanted into a different network?
        + Is the LT in the same basin as the full network?
            + Can we train the same network with and without LT pruning and get LMC between the two?
            + What about different degrees of pruning?
        + What makes a LT and LT?
            + magnitude has something to do with it
            + Is iterative pruning squeezing the information into smaller channels?
                + If so, can we replicate by:
                    + Just Training A Smaller Network? No. (random reinit experiments)
                    + Merging based on correlations before pruning (with less/no iterations)
                        + "One-Shot Merge-Pruning"
                        + Try to squeeze information into smaller channels without re-training
                + do other pruning methods work in fewer iterations?
            + Weight analysis
                + What are weight trajectories
                    + Anything distinct about pruned vs non-pruned weights?
                        + magnitude trajectories
                        + change in magnitude
                        + coorelation/mutual information, etc with other weights
                        + ??
                    


    Masking types from LTH paper:
    + pg 24: Lenet on MNIST (same for Conv-X nets)
        + 60k train, 10k test
        + 24k training iterations in LMC (50k in LTH) 
            + 24 epochs, 10k iterations / epoch
        + Optimizer: Adam
        + Initialization: Glorot
        + Learning Rate: 12e-4 (LMC paper, Table 1)
        + Batch Size: 60 -> 
        + Pruning Rate: 
            + 20% per iteration on most layers
            + 10% for output layers
            + LMC: 3.5%
        + Pruning Groups: Layer-wise, conv-only is best here, but both is ok
        + evaluated training and test every 100 iterations
    + pg 39: VGG and Resnet-20 on CIFAR10 
        + 50k training, 10k test in original
        + with batch size 128, there are 391 iterations per epoch
        + Pruning Groups: Global (tested Layerwise too, said accuracy drops off faster with layer-wise)
        + VGG: (pg 39)
            + Liu 2019 adaptation for CIFAR10
            + Optimizer: SGD
                + LTH: 160 epochs (112,480 iterations (703 iterations/epoch)) with momentum 0.9
                + LMC: 161 epochs, 63k iterations
                + LMC: decreasing LR by a factor of 10 at 80 (81.92?) and 120 (122.88?) epochs
            + Pruning Rate: 
                + 20% per iteration for conv layers
                + 0% on output layer
                + LMC: 1.5% (standard, warmup), 5.5% (low)
            + Batch Size: 64 (128 in Linear Mode Paper)
            + Initialization: Glorot
            + Batch Normalization
            + Weight Decay: 1e-4
            + LR Schedules: 160 epochs (112,480 iterations) total???
                + used k = 5000 linear warm-up???
                + 1e-2 or 1e-1 initial LR with linear warm-up
                + 1e-3 or 1e-2 LR at 80 epochs
                + 1e-4 or 1e-3 LR at 120 epochs
        + ResNet-20: (pg 40)
            + He et al. 2016
            + Optimizer: SGD
                + 30k iterations with momentum 0.9
                + decreasing LR by 10x at 20k and 25k iterations
                + LR 1e-1 and 1e-2 (1e-2 worked better....)
            + Pruning Rate:
                + 20% per iteration for most conv layers
                + 0% on "parameters used to downsample residual connections"
                + 0% on output layer
                + LMC: 16.8% (standard), 8.6% (low and warmup)
            + 45k training samples, 5k validation samples, separate test set?
            + Augment Training data with:
                + random flips
                + random 4 pixel pads and crops
            + Batch Size: 128
            + Batch Normalization
            + Weight Decay: 1e-4
            + LR Schedule: (say 1e-1 is typically used, but LTH didn't work with this high LR)
                + LMC: decreasing LR by a factor of 10 at 80 (81.92?) and 120 (122.88?) epochs
                + used k=20000 linear warm-up
    + LMC: ResNet-50 and Inception-v3 on ImageNet
        + ResNet-50
            + 90 epochs
            + Batch Size: 1024
            + Optimizer: SGD + Momentum
                + LR: 5 epoch warmup to 0.4, 10x drop at 30, 60, 80 epochs
            + Batch Normalization
            + Pruning: 30% ONE SHOT PRUNING
        + Inception-v3
            + 171 epochs
            + Batch Size: 1024
            + Optimizer: SGD + Momentum
                + LR: linear decay from 3e-2 to 5e-3
            + Batch Normalization
            + Pruning: 30% ONE SHOT PRUNING
                
                



    + don't prune last layer
    + global and layer-wise pruning (start w/global)
    + no pruning of residual projection (conv) layers (ResNet20)
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
            history = {}
            for iteration_n in range(self.num_pruning_iterations_N):
                # 5: Train m ⊙ Wk to m ⊙ WT with noise u ′∼ U:WT = Ak→T(m ⊙ Wk, u′).
                
                early_stopping = self._make_early_stopping_callback()
                model_history = self._fit_model(
                    self.fit,
                    dataset,
                    model,
                    [early_stopping],
                    epochs=self.epochs_per_iteration,
                )
                
                # TODO: mask num free parameters
                # TODO: rewinding...? Marking what epoch was used?
                self._accumulate_model_history(
                    history,
                    model_history,
                    model.network.num_free_parameters,
                    early_stopping,
                )

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

    def prune_network(self, model, proportion:float):
        # by layer or entire network (or by group?)
        # pruning rule: |weight|
        # pruning proportion 

