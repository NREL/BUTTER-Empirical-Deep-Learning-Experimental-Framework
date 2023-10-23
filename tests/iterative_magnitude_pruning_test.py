import math
import sys

import numpy
from dmp.context import Context


from dmp.model.named.lenet import Lenet
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

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

from dmp.marshaling import marshal

import time

import jobqueue
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue

import sys

from dmp.marshaling import marshal
from pprint import pprint
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.model.dense_by_size import DenseBySize
from dmp.layer.dense import Dense
from dmp.dataset.dataset_spec import DatasetSpec
import pytest
import dmp.jobqueue_interface.worker
import tensorflow
import sys
from jobqueue.connect import load_credentials

from jobqueue.job import Job
import numpy
import pandas
from tensorflow.python.framework.ops import re

# from dmp import jobqueue_interface
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)
from dmp.task.experiment.pruning_experiment.iterative_pruning_experiment import (
    IterativePruningExperiment,
)
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

sys.path.insert(0, "./")


# strategy = dmp.jobqueue_interface.worker.make_strategy(None, [0], 1024*12)
strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
credentials = load_credentials("dmp")
schema = PostgresSchema(credentials)
worker = Worker(
    None,
    schema,
    None,
    strategy,
    {},
)  # type: ignore


def run_experiment(run):
    context = Context(worker, Job(), run)
    run(context)


def test_pruning_experiment():
    run = Run(
        experiment=IterativePruningExperiment(
            data={
                "test": True,
            },
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
                        "kernel_constraint": keras_kwcfg(
                            "ParameterMask",
                        ),
                    },
                ),
            ),
            fit={
                "batch_size": 60,
                "epochs": 32,
            },
            optimizer={
                "class": "Adam",
                "learning_rate": 12e-4,
            },
            loss=None,
            early_stopping=keras_kwcfg(
                "DMPEarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=10,
                restore_best_weights=True,
            ),
            pruning=PruningConfig(
                iterations=3,
                method=MagnitudePruner(0.1),
                max_epochs_per_iteration=32,
                rewind_epoch=TrainingEpoch(
                    epoch=0,
                    fit_number=0,
                    fit_epoch=0,
                ),
                rewind_optimizer=True,
                new_seed=False,
            ),
        ),
        run=RunSpec(
            seed=0,
            data={},
            record_post_training_metrics=True,
            record_times=True,
            model_saving=ModelSavingSpec(
                save_initial_model=True,
                save_trained_model=True,
                save_fit_epochs=[],
                save_epochs=[],
                fixed_interval=1,
                fixed_threshold=4,
                exponential_rate=math.pow(2, 1 / 2.0),
            ),
            saved_models=[],
            resume_checkpoint=None,
        ),
    )

    run_experiment(run)


# def test_pruning_experiment():
#     experiment = IterativePruningExperiment(
#         seed=0,
#         batch="test",
#         experiment_tags={
#             "simple": True,
#         },
#         run_tags={
#             "test": True,
#         },
#         precision="float32",
#         dataset=DatasetSpec(
#             # 'titanic',
#             # 'pmlb',
#             "GaussianClassificationDataset_2_10_100",
#             # # 'GaussianRegressionDataset_20_100',
#             "synthetic",
#             "shuffled_train_test_split",
#             0.2,
#             0.05,
#             0.0,
#         ),
#         model=DenseBySize(
#             input=None,
#             output=None,
#             shape="rectangle",
#             size=16384,
#             depth=4,
#             search_method="integer",
#             inner=Dense.make(
#                 -1,
#                 {
#                     "activation": "relu",
#                     "kernel_initializer": "GlorotUniform",
#                     "kernel_constraint": keras_kwcfg(
#                         "ParameterMask",
#                     ),
#                 },
#             ),
#         ),
#         fit={
#             "batch_size": 256,
#             "epochs": 5,
#         },
#         optimizer={
#             "class": "SGD",
#             "learning_rate": 0.01,
#             "momentum": 0.9,
#         },
#         loss=None,
#         early_stopping=keras_kwcfg(
#             "EarlyStopping",
#             monitor="val_loss",
#             min_delta=0,
#             patience=1,
#             restore_best_weights=True,
#         ),
#         record=RunSpec(
#             post_training_metrics=True,
#             times=True,
#             model=None,
#             metrics=None,
#             model_saving=ModelSavingConfig(
#                 save_initial_model=True,
#                 save_trained_model=True,
#                 save_epochs=[],
#                 save_model_epochs=[],
#                 fixed_interval=1,
#                 fixed_threshold=32,
#                 exponential_rate=2,
#             ),
#             resume_checkpoint=None,
#         ),
#         num_pruning_iterations=4,
#         pre_prune_epochs=2,
#         pre_pruning_trigger=None,
#         pruning_method=MagnitudePruner(
#             pruning_rate=1.0 - 0.5 ** (1 / 4),
#         ),
#         pruning_trigger=None,
#         max_pruning_epochs=5,
#         rewind_point=True,
#     )

#     run_experiment(experiment)


if __name__ == "__main__":
    # test_growth_experiment()
    # test_simple()
    # test_mnist()
    # test_mnist_lenet()
    # test_from_optimizer()
    # test_get_sizes()
    # test_growth_experiment_mnist()
    # test_imagenet16()
    test_pruning_experiment()
