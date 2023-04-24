import sys

from jobqueue.job import Job
import numpy
import pandas
from tensorflow.python.framework.ops import re
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense_conv import DenseConv, conv_3x3

# from dmp import jobqueue_interface
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.sequential_model import SequentialModel
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
    ExperimentRecordSettings,
)
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

sys.path.insert(0, './')

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

from dmp.marshaling import marshal

# strategy = dmp.jobqueue_interface.worker.make_strategy(None, [0], 1024*12)
strategy = dmp.jobqueue_interface.worker.make_strategy(6, None, None)
worker = Worker(
    None,
    None,
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


def test_vgg16():
    conv_config = {
        'padding': 'same',
        'use_bias': False,
    }
    experiment = TrainingExperiment(
        seed=0,
        batch='test',
        tags={
            'model_family': 'vgg',
            'model_name': 'vgg16',
        },
        run_tags={
            'test': True,
        },
        precision='float32',
        dataset=DatasetSpec(
            # 'mnist',
            # 'keras',
            'cifar10',
            'keras',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        
        # RETHINKING THE VALUE OF NETWORK PRUNING: https://arxiv.org/pdf/1810.05270.pdf
        # reference implementation: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/lottery-ticket/l1-norm-pruning/models/vgg.py
        # Original VGG: https://arxiv.org/pdf/1409.1556.pdf
        model=SequentialModel(
            [
                DenseConv.make(64, [3, 3], [1, 1], conv_config),
                DenseConv.make(64, [3, 3], [1, 1], conv_config),
                MaxPool.make([2, 2], [2, 2]),
                DenseConv.make(128, [3, 3], [1, 1], conv_config),
                DenseConv.make(128, [3, 3], [1, 1], conv_config),
                MaxPool.make([2, 2], [2, 2]),
                DenseConv.make(256, [3, 3], [1, 1], conv_config),
                DenseConv.make(256, [3, 3], [1, 1], conv_config),
                DenseConv.make(256, [1, 1], [1, 1], conv_config),
                MaxPool.make([2, 2], [2, 2]),
                DenseConv.make(512, [3, 3], [1, 1], conv_config),
                DenseConv.make(512, [3, 3], [1, 1], conv_config),
                DenseConv.make(512, [1, 1], [1, 1], conv_config),
                MaxPool.make([2, 2], [2, 2]),
                DenseConv.make(512, [3, 3], [1, 1], conv_config),
                DenseConv.make(512, [3, 3], [1, 1], conv_config),
                DenseConv.make(512, [3, 3], [1, 1], conv_config),
                AvgPool.make([2, 2], [2, 2]), # MaxPool in original paper
                Flatten(),
                Dense.make(512),
                Dense.make(512),
            ]
        ),
        fit={
            'batch_size': 256,
            'epochs': 1,
        },
        optimizer={'class': 'Adam', 'learning_rate': 0.0001},
        loss=None,
        early_stopping=make_keras_kwcfg(
            'EarlyStopping',
            monitor='val_loss',
            min_delta=0,
            patience=50,
            restore_best_weights=True,
        ),
        record=ExperimentRecordSettings(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
        ),
    )

    run_experiment(experiment)


if __name__ == '__main__':
    test_vgg16()
