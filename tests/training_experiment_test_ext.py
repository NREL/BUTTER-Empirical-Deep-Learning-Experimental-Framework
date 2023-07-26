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


# strategy = dmp.jobqueue_interface.worker.make_strategy(None, [0], 1024*12)
strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
worker = Worker(
    None,
    None,
    None,
    strategy,
    {},
)  # type: ignore


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


def test_simple():
    experiment = TrainingExperiment(
        seed=0,
        batch="test",
        experiment_tags={"simple": True},
        run_tags={"test": True},
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
            "batch_size": 256,
            "epochs": 5,
        },
        optimizer={
            "class": "Adam",
            "learning_rate": 0.001,
        },
        loss=None,
        early_stopping=None,
        record=RunSpec(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
            resume_from=None,
        ),
    )

    run_experiment(experiment)


def test_mnist():
    width = int(2**4)

    experiment = TrainingExperiment(
        seed=0,
        batch="test",
        experiment_tags={"simple": True},
        run_tags={"test": True},
        precision="float32",
        dataset=DatasetSpec(
            name="mnist",
            source="keras",
            method="shuffled_train_test_split",
            test_split=0.2,
            validation_split=0.05,
            label_noise=0.0,
        ),
        model=CNNStack(
            input=None,
            output=None,
            num_stacks=3,
            cells_per_stack=1,
            stem="conv_5x5_1x1_same",
            downsample="max_pool_2x2_2x2_valid",
            cell="conv_5x5_1x1_same",
            final=FullyConnectedNetwork(
                input=None,
                output=None,
                widths=[width * 2, width * 2],
                residual_mode="none",
                flatten_input=True,
                inner=Dense.make(-1, {}),
            ),
            stem_width=width,
            stack_width_scale_factor=1.0,
            downsample_width_scale_factor=1.0,
            cell_width_scale_factor=1.0,
        ),
        fit={
            "batch_size": 256,
            "epochs": 1,
        },
        optimizer={"class": "Adam", "learning_rate": 0.0001},
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
            resume_from=None,
        ),
    )

    run_experiment(experiment)


def test_mnist_lenet():
    experiment = TrainingExperiment(
        seed=0,
        batch="test",
        experiment_tags={
            "model_family": "lenet",
            "model_name": "lenet_relu",
        },
        run_tags={
            "test": True,
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
            "batch_size": 256,
            "epochs": 1,
        },
        optimizer={"class": "Adam", "learning_rate": 0.0001},
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
            resume_from=None,
        ),
    )

    run_experiment(experiment)


def test_growth_experiment():
    experiment = GrowthExperiment(
        seed=0,
        batch="test",
        experiment_tags={
            "simple": True,
        },
        run_tags={"test": True},
        precision="float32",
        dataset=DatasetSpec(
            "titanic",
            "pmlb",
            "shuffled_train_test_split",
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            input=None,
            output=None,
            shape="rectangle",
            size=4096,
            depth=3,
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
        optimizer={"class": "Adam", "learning_rate": 0.001},
        loss=None,
        early_stopping=None,
        record=RunSpec(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
            resume_from=None,
        ),
        growth_trigger=keras_kwcfg(
            "ProportionalStopping",
            restore_best_weights=True,
            monitor="val_loss",
            min_delta=0.005,
            patience=3,
            verbose=1,
            mode="min",
            baseline=None,
            # start_from_epoch=0,
        ),
        # growth_trigger=None,
        scaling_method=WidthScaler(),
        transfer_method=OverlayTransfer(),
        growth_scale=2.0,
        initial_size=4,
        max_epochs_per_stage=300,
        max_equivalent_epoch_budget=1000,
    )

    run_experiment(experiment)


def test_growth_experiment_mnist():
    width = 2
    batch_size = 64
    optimizer = {
        "class": "Adam",
        "learning_rate": 0.001,
    }

    experiment = GrowthExperiment(
        seed=0,
        batch="test",
        experiment_tags={
            "simple": True,
        },
        run_tags={
            "test": True,
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
            num_stacks=3,
            cells_per_stack=1,
            stem="conv_5x5_1x1_same",
            downsample="max_pool_2x2_2x2_valid",
            cell="conv_5x5_1x1_same",
            final=FullyConnectedNetwork(
                input=None,
                output=None,
                widths=[width * 2, width * 2],
                residual_mode="none",
                flatten_input=True,
                inner=Dense.make(-1, {}),
            ),
            stem_width=width,
            stack_width_scale_factor=1.0,
            downsample_width_scale_factor=1.0,
            cell_width_scale_factor=1.0,
        ),
        fit={
            "batch_size": batch_size,
            "epochs": 1024 * 32,
        },
        optimizer=optimizer,
        loss=None,
        early_stopping=keras_kwcfg(
            "EarlyStopping",
            monitor="val_loss",
            min_delta=0,
            patience=16,
            restore_best_weights=True,
        ),
        record=RunSpec(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
            resume_from=None,
        ),
        growth_trigger=keras_kwcfg(
            "ProportionalStopping",
            restore_best_weights=True,
            monitor="val_loss",
            min_delta=0.01,
            patience=0,
            verbose=1,
            mode="min",
            baseline=None,
            # start_from_epoch=0,
        ),
        scaling_method=WidthScaler(),
        transfer_method=OverlayTransfer(),
        growth_scale=2.0,
        initial_size=16,
        # max_epochs_per_stage=1024 * 2,
        max_epochs_per_stage=1,
        max_equivalent_epoch_budget=2048,
    )

    run_experiment(experiment)


def test_from_optimizer():
    """





    'activation','relu'
    'activity_regularizer',
    'batch','optimizer_energy_1_cpu'
    'batch_size',128
    'bias_regularizer',
    'dataset','mnist'
    'depth',2
    'early_stopping',
    'epochs',3000
    'input_activation','relu'
    'kernel_regularizer',
    'label_noise',0
    'learning_rate',0.00001
    'optimizer','SGD'
    'optimizer.config.momentum',0.9
    'optimizer.config.nesterov',false
    'output_activation','softmax'
    'python_version','3.9.10'
    'run_config.shuffle',true
    'shape','rectangle'
    'size',131072
    'task','AspectTestTask'
    'task_version',3
    'tensorflow_version','2.8.0'
    'test_split',0.2
    'test_split_method','shuffled_train_test_split'

    {
      '': 'AspectTestTask',
      'seed': 1666301679,
      'run_config': {
        'verbose': 0,
      },
      'save_every_epochs': null,
    }
    """
    experiment = TrainingExperiment(
        seed=0,
        batch="test",
        experiment_tags=None,
        run_tags={"test": True},
        precision="float32",
        dataset=DatasetSpec(
            "banana",
            "pmlb",
            "shuffled_train_test_split",
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            None,
            None,
            "exponential",
            16384,
            3,
            "integer",
            Dense.make(
                -1,
                {
                    "activation": "relu",
                    "kernel_initializer": "GlorotUniform",
                },
            ),
        ),
        fit={
            "batch_size": 16,
            "epochs": 3,
        },
        optimizer={"class": "Adam", "learning_rate": 0.0001},
        loss=None,
        early_stopping=None,
        record=RunSpec(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
            resume_from=None,
        ),
    )

    run_experiment(experiment)


def test_imagenet16():
    width = int(2**4)

    experiment = TrainingExperiment(
        seed=0,
        batch="test",
        experiment_tags={"simple": True},
        run_tags={"test": True},
        precision="float32",
        dataset=DatasetSpec(
            "imagenet_16",
            "imagenet",
            "shuffled_train_test_split",
            0.2,
            0.05,
            0.0,
        ),
        model=CNNStack(
            input=None,
            output=None,
            num_stacks=2,
            cells_per_stack=2,
            stem="conv_3x3_1x1_valid",
            downsample="max_pool_2x2_2x2_same",
            cell="conv_3x3_1x1_same",
            final=FullyConnectedNetwork(
                input=None,
                output=None,
                widths=[width * 2, width * 2],
                residual_mode="none",
                flatten_input=True,
                inner=Dense.make(-1, {}),
            ),
            stem_width=width,
            stack_width_scale_factor=1.0,
            downsample_width_scale_factor=1.0,
            cell_width_scale_factor=1.0,
        ),
        fit={
            "batch_size": 256,
            "epochs": 1,
        },
        optimizer={"class": "Adam", "learning_rate": 0.0001},
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
            resume_from=None,
        ),
    )

    run_experiment(experiment)


def test_pruning_experiment():
    experiment = IterativePruningExperiment(
        seed=0,
        batch="test",
        experiment_tags={"simple": True},
        run_tags={"test": True},
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
            "batch_size": 256,
            "epochs": 5,
        },
        optimizer={
            "class": "SGD",
            "learning_rate": 0.001,
            "momentum": 0.9,
        },
        loss=None,
        early_stopping=keras_kwcfg(
            "EarlyStopping",
            monitor="val_loss",
            min_delta=0,
            patience=1,
            restore_best_weights=True,
        ),
        record=RunSpec(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
            resume_from=None,
        ),
        num_pruning_iterations=4,
        pre_prune_epochs=2,
        pre_pruning_trigger=None,
        pruning_method=MagnitudePruner(
            pruning_rate=1.0 - 0.5 ** (1 / 4),
        ),
        pruning_trigger=None,
        max_pruning_epochs=5,
        rewind_point=True,
    )

    run_experiment(experiment)


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
