from dataclasses import dataclass
import sys
from typing import List, Union

from jobqueue.job import Job
import numpy

from dmp.model.named.resenet20 import Resnet20
from dmp.task.experiment.lth.lth_experiment import LTHExperiment
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.run import Run
from dmp.keras_interface.keras_utils import keras_kwcfg

sys.path.insert(0, "./")


from dmp.dataset.dataset_spec import DatasetSpec
from pprint import pprint
from paper_param import get_paper_param
from dmp.marshaling import marshal

import tests.experiment_test_util as experiment_test_util

param = get_paper_param("Linear_Mode_Connectivity", "RESNET", "Low")


def test_resenet20():
    # ResNet Paper: https://arxiv.org/pdf/1512.03385.pdf
    # One imperfect implementation: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/2ac473d70a09810df888e932bb394f225f9ed2d1/cifar/lottery-ticket/l1-norm-pruning/models/resnet.py#L20
    # Another not-quite-right implementation: https://github.com/LuigiRussoDev/ResNets/blob/master/resnet20.py
    # Helpful reference: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
    # Helpful but incomplete diagram: https://www.researchgate.net/figure/ResNet-20-architecture_fig3_351046093

    # cells_per_stack = 3
    # depth = cells_per_stack * 3 + 2
    # def make_downsample(input):
    #     return AvgPool.make(
    #         [2, 2],
    #         [2, 2],
    #         {'padding': 'same'},
    #         input,
    #     )
    pruning_target = 0.001
    base_pruning_rate = param["pruning_rate"]
    base_survival_rate = 1.0 - base_pruning_rate

    rewind_epoch = 1
    pruning_configs = []
    for survival_rate in [
        base_survival_rate**4,
        base_survival_rate**2,
        base_survival_rate,
        base_survival_rate ** (1 / 2),
        base_survival_rate ** (1 / 4),
    ]:
        pruning_iterations = int(
            numpy.ceil(numpy.log(pruning_target) / numpy.log(survival_rate))
        )
        pruning_rate = 1.0 - survival_rate

        pruning_config = PruningConfig(
            iterations=pruning_iterations,
            method=MagnitudePruner(pruning_rate),
            max_epochs_per_iteration=32,
            rewind_epoch=TrainingEpoch(
                epoch=rewind_epoch,
                model_number=0,
                model_epoch=rewind_epoch,
            ),
            rewind_optimizer=True,
            new_seed=False,
        )

        for rewind_epoch in [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            12,
            16,
            20,
            24,
            32,
            48,
            64,
            96,
            128,
        ]:
            pruning_configs.append(
                PruningConfig(
                    iterations=pruning_iterations,
                    method=MagnitudePruner(pruning_rate),
                    max_epochs_per_iteration=32,
                    rewind_epoch=TrainingEpoch(
                        epoch=rewind_epoch,
                        model_number=0,
                        model_epoch=rewind_epoch,
                    ),
                    rewind_optimizer=True,
                    new_seed=False,
                )
            )
            pruning_configs.append(pruning_config)

    seed = 1234

    run = Run(
        experiment=LTHExperiment(
            data={
                "lth": True,
                "batch": "lth_resnet_low_1",
                "group": "lth_resnet_low",
                "model_family": "resnet",
                "model_name": "Resnet20",
                "resnet_depth": 20,
            },
            precision="float32",
            dataset=DatasetSpec(
                "cifar10",
                "keras",
                "shuffled_train_test_split",
                0.2,
                0.05,
                0.0,
            ),
            model=Resnet20(),
            fit={
                "batch_size": param["batch"],
                "epochs": int(
                    param["batch"] * param["train_Step"] // 60000
                ),  # 60000 is the number of training images in CIFAR10
            },
            optimizer={
                "class": param["optimizer"],
                "learning_rate": param["learning_rate"],
            },
            loss=None,
            early_stopping=keras_kwcfg(
                "EarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=50,
                restore_best_weights=True,
            ),
            pruning_configs=pruning_configs,
            num_additional_seeds_per_config=1,
        ),
        run=RunSpec(
            seed=seed,
            data={},
            record_post_training_metrics=True,
            record_times=True,
            model_saving=ModelSavingSpec(
                save_initial_model=True,
                save_trained_model=True,
                save_model_epochs=[],
                save_epochs=[],
                fixed_interval=1,
                fixed_threshold=-1,
                exponential_rate=0,
            ),
            resume_checkpoint=None,
        ),
    )

    experiment_test_util.run_experiment(run)


if __name__ == "__main__":
    test_resenet20()
