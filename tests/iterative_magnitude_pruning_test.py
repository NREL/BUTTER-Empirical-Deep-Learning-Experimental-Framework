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
from dmp.task.experiment.training_experiment.experiment_record_settings import (
    RunSpecificConfig,
)
from dmp.task.experiment.training_experiment.hybrid_save_mode import HybridSaveMode
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

sys.path.insert(0, './')


# strategy = dmp.jobqueue_interface.worker.make_strategy(None, [0], 1024*12)
strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
credentials = load_credentials('dmp')
schema = PostgresSchema(credentials)
worker = Worker(
    None,
    schema,
    None,
    strategy,
    {},
)  # type: ignore


def run_experiment(experiment):
    results = experiment(worker, Job())
    print('experiment_attrs\n')
    pprint(results.experiment_attrs)
    print('experiment_tags\n')
    pprint(results.experiment_tags)
    print('run_data\n', results.run_data)
    print('run_history\n', results.run_history)
    print('run_extended_history\n', results.run_extended_history)
    return results


def test_pruning_experiment():
    experiment = IterativePruningExperiment(
        seed=0,
        batch='test',
        tags={
            'simple': True,
        },
        run_tags={
            'test': True,
        },
        precision='float32',
        dataset=DatasetSpec(
            # 'titanic',
            # 'pmlb',
            'GaussianClassificationDataset_2_10_100',
            # # 'GaussianRegressionDataset_20_100',
            'synthetic',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            input=None,
            output=None,
            shape='rectangle',
            size=16384,
            depth=4,
            search_method='integer',
            inner=Dense.make(
                -1,
                {
                    'activation': 'relu',
                    'kernel_initializer': 'GlorotUniform',
                    'kernel_constraint': make_keras_kwcfg(
                        'ParameterMask',
                    ),
                },
            ),
        ),
        fit={
            'batch_size': 256,
            'epochs': 5,
        },
        optimizer={
            'class': 'SGD',
            'learning_rate': 0.01,
            'momentum': 0.9,
        },
        loss=None,
        early_stopping=make_keras_kwcfg(
            'EarlyStopping',
            monitor='val_loss',
            min_delta=0,
            patience=1,
            restore_best_weights=True,
        ),
        record=RunSpecificConfig(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
            model_saving=HybridSaveMode(
                save_initial_model=True,
                save_trained_model=True,
                save_epochs=[],
                save_model_epochs=[],
                fixed_interval=1,
                fixed_threshold=32,
                exponential_rate=2,
            ),
            resume_from=None,
        ),
        num_pruning_iterations=4,
        pre_prune_epochs=2,
        pre_pruning_trigger=None,
        pruning_method=MagnitudePruner(
            prune_percent=1.0 - 0.5 ** (1 / 4),
        ),
        pruning_trigger=None,
        max_pruning_epochs=5,
        rewind=True,

    )

    run_experiment(experiment)


if __name__ == '__main__':
    # test_growth_experiment()
    # test_simple()
    # test_mnist()
    # test_mnist_lenet()
    # test_from_optimizer()
    # test_get_sizes()
    # test_growth_experiment_mnist()
    # test_imagenet16()
    test_pruning_experiment()
