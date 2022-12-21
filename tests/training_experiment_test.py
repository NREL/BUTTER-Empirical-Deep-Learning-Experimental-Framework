import sys

from dmp import jobqueue_interface
from dmp.layer.visitor.keras_interface.keras_utils import make_keras_kwcfg
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.growth_experiment.growth_method.overlay_growth_method import OverlayGrowthMethod

sys.path.insert(0, './')

import tensorflow
from dmp.worker import Worker
import pytest

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.layer.dense import Dense
from dmp.model.dense_by_size import DenseBySize

from dmp.task.training_experiment.training_experiment import TrainingExperiment
from pprint import pprint


def test_simple():
    experiment = TrainingExperiment(
        seed=0,
        batch='test',
        dataset=DatasetSpec(
            'banana',
            'pmlb',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            None,
            None,
            'exponential',
            16384,
            3,
            Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit_config={
            'batch_size': 16,
            'epochs': 5,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.0001
        },
        loss=None,
        early_stopping=None,
        save_every_epochs=-1)

    worker = Worker(
        None,
        None,
        tensorflow.distribute.get_strategy(),
        {},
    )

    results = experiment(worker)
    pprint(jobqueue_interface.jobqueue_marshal.marshal(results), indent=2)


def test_growth_experiment():
    experiment = GrowthExperiment(
        seed=0,
        batch='test',
        dataset=DatasetSpec(
            'titanic',
            'pmlb',
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
            depth=3,
            inner=Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit_config={
            'batch_size': 32,
            'epochs': 4,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.001
        },
        loss=None,
        early_stopping=None,
        save_every_epochs=-1,
        growth_trigger=make_keras_kwcfg(
            'EarlyStopping',
            restore_best_weights=True,
            monitor='val_loss',
            min_delta=.01,
            patience=0,
            verbose=0,
            mode='min',
            baseline=None,
            # start_from_epoch=0,
        ),
        growth_method=OverlayGrowthMethod(),
        growth_scale=2.0,
        initial_size=128,
        max_total_epochs=16,
        max_equivalent_epoch_budget=16,
    )

    worker = Worker(
        None,
        None,
        tensorflow.distribute.get_strategy(),
        {},
    )

    results = experiment(worker)
    pprint(jobqueue_interface.jobqueue_marshal.marshal(results), indent=2)


test_growth_experiment()