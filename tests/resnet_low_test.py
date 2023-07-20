from dataclasses import dataclass
import sys
from typing import List, Union

from jobqueue.job import Job
import numpy
import pandas
from tensorflow.python.framework.ops import re
from dmp.layer.add import Add
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense_conv import DenseConv

# from dmp import jobqueue_interface
from dmp.layer.flatten import Flatten
from dmp.layer.global_average_pooling import GlobalAveragePooling
from dmp.layer.identity import Identity
from dmp.layer.layer import Layer, LayerConfig, LayerFactory
from dmp.layer.max_pool import MaxPool
from dmp.layer.op_layer import OpLayer
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.layer_factory_model import LayerFactoryModel
from dmp.structure.res_net_block import ResNetBlock
from dmp.structure.sequential_model import SequentialModel
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.structure.batch_norm_block import BatchNormBlock
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)
from dmp.task.experiment.pruning_experiment.iterative_pruning_experiment import (
    IterativePruningExperiment,
)
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)
from dmp.task.experiment.run_spec import (
    RunSpec,
)
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

sys.path.insert(0, "./")

import tensorflow
import dmp.jobqueue_interface.worker
import pytest

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.layer.dense import Dense
from dmp.model.dense_by_size import DenseBySize

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from pprint import pprint
from paper_param import get_paper_param
from dmp.marshaling import marshal

# strategy = dmp.jobqueue_interface.worker.make_strategy(None, [0], 1024*12)
# strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
strategy = tensorflow.distribute.get_strategy()
worker = Worker(
    None,
    None,
    None,
    strategy,
    {},
)  # type: ignore
param = get_paper_param("Linear_Mode_Connectivity", "RESNET", "Low")


def run_experiment(experiment):
    results = experiment(worker, Job())
    print("experiment_attrs\n")
    pprint(results.experiment_attrs)
    print("experiment_tags\n")
    pprint(results.experiment_tags)
    print("run_data\n", results.run_data)
    print("run_history\n", results.run_history)
    print("run_extended_history\n", results.run_extended_history)
    return results


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

    model = LayerFactoryModel(
        layer_factory=SequentialModel(
            [
                BatchNormBlock(
                    DenseConv.make(
                        16,
                        [7, 7],
                        [2, 2],
                        {
                            "padding": "same",
                            "use_bias": False,
                        },
                    )
                ),
                ResNetBlock(16, 1),
                ResNetBlock(16, 1),
                ResNetBlock(16, 1),
                ResNetBlock(32, 2),
                ResNetBlock(32, 1),
                ResNetBlock(32, 1),
                ResNetBlock(64, 2),
                ResNetBlock(64, 1),
                ResNetBlock(64, 1),
                GlobalAveragePooling(),
                Flatten(),
            ]
        )
    )

    experiment = TrainingExperiment(
        seed=0,
        batch="test",
        experiment_tags={
            "model_family": "resnet",
            "model_name": f"resnet20",
            "resnet_depth": 20,
        },
        run_tags={
            "test": True,
        },
        precision="float32",
        dataset=DatasetSpec(
            # 'mnist',
            # 'keras',
            "cifar10",
            "keras",
            "shuffled_train_test_split",
            0.2,
            0.05,
            0.0,
        ),
        model=model,
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
        record=RunSpec(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
        ),
    )

    run_experiment(experiment)


if __name__ == "__main__":
    test_resenet20()
