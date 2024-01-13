from dmp.context import Context
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.marshaling import marshal
from pprint import pprint
from dmp.task.experiment.training_experiment.run_spec import RunConfig
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.model.dense_by_size import DenseBySize
from dmp.layer.dense import Dense
from dmp.dataset.dataset_spec import DatasetSpec
import pytest
import dmp.script.worker
import tensorflow
import sys

from jobqueue.job import Job
import numpy
import pandas

# from dmp import script
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)
from dmp.task.run import Run
from dmp.worker import Worker

import tests.experiment_test_util as experiment_test_util


def test_simple():
    run = Run(
        experiment=TrainingExperiment(
            data={"test": True},
            precision="float32",
            dataset=DatasetSpec(
                # 'titanic',
                # 'pmlb',
                "GaussianClassificationDataset_2_10_100",
                # # 'GaussianRegressionDataset_20_100',
                "synthetic",
                "shuffled_train_test_split",
                0.2,
                0.05,
                0.0,
            ),
            model=DenseBySize(
                input=None,
                output=None,
                shape="rectangle",
                size=16384,
                depth=4,
                search_method="integer",
                inner=Dense.make(
                    -1,
                    {
                        "activation": "relu",
                        "kernel_initializer": "GlorotUniform",
                    },
                ),
            ),
            fit={
                "batch_size": 32,
                "epochs": 100,
            },
            optimizer={
                "class": "Adam",
                "learning_rate": 0.01,
            },
            loss=None,
            early_stopping=keras_kwcfg(
                "EarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=5,
                restore_best_weights=True,
            ),
        ),
        config=RunConfig(
            seed=0,
            data={
                "test": True,
            },
            record_post_training_metrics=True,
            record_times=True,
            model_saving=None,
            resume_checkpoint=None,
        ),
    )

    experiment_test_util.run_experiment(run)


def test_mnist_lenet():
    run = Run(
        experiment=TrainingExperiment(
            data={
                "test": True,
                "model_family": "lenet",
                "model_name": "lenet_relu",
            },
            precision="float32",
            dataset=DatasetSpec(
                "mnist",
                "keras",
                "shuffled_train_test_split",
                0.2,
                0.05,
                0.0,
            ),
            model=CNNStack(
                input=None,
                output=None,
                num_stacks=2,
                cells_per_stack=1,
                stem="conv_5x5_1x1_same",
                downsample="max_pool_2x2_2x2_valid",
                cell="conv_5x5_1x1_valid",
                final=FullyConnectedNetwork(
                    input=None,
                    output=None,
                    widths=[120, 84],
                    residual_mode="none",
                    flatten_input=True,
                    inner=Dense.make(-1, {}),
                ),
                stem_width=6,
                stack_width_scale_factor=16.0 / 6.0,
                downsample_width_scale_factor=1.0,
                cell_width_scale_factor=1.0,
            ),
            fit={
                "batch_size": 128,
                "epochs": 15,
            },
            optimizer={
                "class": "Adam",
                "learning_rate": 0.01,
            },
            loss=None,
            early_stopping=keras_kwcfg(
                "EarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=2,
                restore_best_weights=True,
            ),
        ),
        config=RunConfig(
            seed=0,
            data={
                "test": True,
            },
            record_post_training_metrics=False,
            record_times=True,
            model_saving=None,
            resume_checkpoint=None,
        ),
    )

    experiment_test_util.run_experiment(run)


if __name__ == "__main__":
    # test_simple()
    test_mnist_lenet()
