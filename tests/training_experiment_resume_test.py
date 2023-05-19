import sys
import uuid
from jobqueue.connect import load_credentials

from jobqueue.job import Job
import numpy
import pandas
from tensorflow.python.framework.ops import re
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense_conv import DenseConv

# from dmp import jobqueue_interface
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.layer_factory_model import LayerFactoryModel
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.structure.batch_norm_block import BatchNormBlock
from dmp.structure.sequential_model import SequentialModel
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
from dmp.task.experiment.training_experiment.hybrid_save_mode import HybridSaveMode
from dmp.task.experiment.training_experiment.model_state_resume_config import (
    ModelStateResumeConfig,
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

credentials = load_credentials('dmp')
schema = PostgresSchema(credentials)
strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
worker = Worker(
    None,
    schema,
    None,
    strategy,
    {},
)  # type: ignore

# 
# worker = Worker(
#     None,
#     None,
#     None,
#     strategy,
#     {},
# )  # type: ignore


def run_experiment(experiment, job_id=None):
    job = Job()
    if job_id is not None:
        job.id = job_id

    results = experiment(worker, job)

    print('experiment_attrs\n')
    pprint(results.experiment_attrs)
    print('experiment_tags\n')
    pprint(results.experiment_tags)
    print('run_data\n', results.run_data)
    print('run_history\n', results.run_history)
    print('run_extended_history\n', results.run_extended_history)
    return results


def test_mnist_lenet():
    experiment = TrainingExperiment(
        seed=100,
        batch='test',
        tags={
            'model_family': 'lenet',
            'model_name': 'lenet_relu',
        },
        run_tags={
            'test': True,
        },
        precision='float32',
        dataset=DatasetSpec(
            'mnist',
            'keras',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=LayerFactoryModel(
            layer_factory=SequentialModel(
                [
                    DenseConv.make(
                        6,
                        [5, 5],
                        [1, 1],
                        {
                            'padding': 'same',
                            'use_bias': True,
                            'activation': 'sigmoid',
                            'kernel_constraint': make_keras_kwcfg('ParameterMask'),
                        },
                    ),
                    AvgPool.make([2, 2], [2, 2]),
                    DenseConv.make(
                        6,
                        [5, 5],
                        [1, 1],
                        {
                            'padding': 'valid',
                            'use_bias': True,
                            'kernel_constraint': make_keras_kwcfg('ParameterMask'),
                        },
                    ),
                    AvgPool.make([2, 2], [2, 2]),
                    FullyConnectedNetwork(
                        widths=[120, 84],
                        flatten_input=True,
                        inner=Dense.make(-1, {}),
                    ),
                ]
            )
        ),
        fit={
            'batch_size': 32,
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
            model_saving=HybridSaveMode(
                save_initial_model=False,
                save_trained_model=False,
                save_model_epochs=[],
                save_epochs=[1],
                fixed_interval=0,
                fixed_threshold=0,
                exponential_rate=0,
                
            ),
        ),
        resume_from=None,
    )

    job_id = uuid.UUID('355d6326-aaf4-4d11-bfbe-d7ae667298f3')
    run_experiment(experiment, job_id=job_id)

def test_resume():
    job_id = uuid.UUID('355d6326-aaf4-4d11-bfbe-d7ae667298f3')
    resume_experiment = experiment = TrainingExperiment(
        seed=100,
        batch='test',
        tags={
            'model_family': 'lenet',
            'model_name': 'lenet_relu',
        },
        run_tags={
            'test': True,
        },
        precision='float32',
        dataset=DatasetSpec(
            'mnist',
            'keras',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=LayerFactoryModel(
            layer_factory=SequentialModel(
                [
                    DenseConv.make(
                        6,
                        [5, 5],
                        [1, 1],
                        {
                            'padding': 'same',
                            'use_bias': True,
                            'activation': 'sigmoid',
                            'kernel_constraint': make_keras_kwcfg('ParameterMask'),
                        },
                    ),
                    AvgPool.make([2, 2], [2, 2]),
                    DenseConv.make(
                        6,
                        [5, 5],
                        [1, 1],
                        {
                            'padding': 'valid',
                            'use_bias': True,
                            'kernel_constraint': make_keras_kwcfg('ParameterMask'),
                        },
                    ),
                    AvgPool.make([2, 2], [2, 2]),
                    FullyConnectedNetwork(
                        widths=[120, 84],
                        flatten_input=True,
                        inner=Dense.make(-1, {}),
                    ),
                ]
            )
        ),
        fit={
            'batch_size': 32,
            'epochs': 10,
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
            model_saving=HybridSaveMode(
                save_initial_model=True,
                save_trained_model=True,
                save_model_epochs=[],
                save_epochs=[],
                fixed_interval=0,
                fixed_threshold=0,
                exponential_rate=0,
                
            ),
        ),
        resume_from=ModelStateResumeConfig(
            run_id=job_id,
            epoch=5,
            model_number=0,
            model_epoch=1,
            load_mask=True,
            load_optimizer=True,
        ),
    )
    run_experiment(resume_experiment)



if __name__ == '__main__':
    test_mnist_lenet()
    test_resume()
