import sys

from dmp import jobqueue_interface
from dmp.task.growth_experiment.scaling_method.width_scaler import WidthScaler
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.growth_experiment.transfer_method.overlay_transfer import OverlayTransfer

sys.path.insert(0, './')

import tensorflow
import dmp.jobqueue_interface.worker
import pytest

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.layer.dense import Dense
from dmp.model.dense_by_size import DenseBySize

from dmp.task.training_experiment.training_experiment import TrainingExperiment
from pprint import pprint

from dmp.marshaling import marshal

strategy = dmp.jobqueue_interface.worker.make_strategy(4, 0, 0, 0)


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
            input=None,
            output=None,
            shape='exponential',
            size=16384,
            depth=3,
            search_method='integer',
            inner=Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit={
            'batch_size': 16,
            'epochs': 5,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.0001
        },
        loss=None,
        early_stopping=None,
        save_every_epochs=-1,
        record_post_training_metrics=True,
        record_times=True,
        )

    worker = Worker(
        None,
        None,
        strategy,
        {},
    )

    results = experiment(worker)
    pprint(marshal.marshal(results), indent=1)


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
            size=4096,
            depth=3,
            search_method='integer',
            inner=Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit={
            'batch_size': 32,
            'epochs': 100,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.001
        },
        loss=None,
        early_stopping=None,
        save_every_epochs=-1,
        record_post_training_metrics=True,
        record_times=True,
        growth_trigger=make_keras_kwcfg(
            'ProportionalStopping',
            restore_best_weights=True,
            monitor='val_loss',
            min_delta=0.005,
            patience=3,
            verbose=1,
            mode='min',
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

    worker = Worker(
        None,
        None,
        strategy,
        {},
    )

    results = experiment(worker)
    pprint(marshal.marshal(results), indent=1)


def test_from_optimizer():

    '''

    

    

"activation","relu"
"activity_regularizer",
"batch","optimizer_energy_1_cpu"
"batch_size",128
"bias_regularizer",
"dataset","mnist"
"depth",2
"early_stopping",
"epochs",3000
"input_activation","relu"
"kernel_regularizer",
"label_noise",0
"learning_rate",0.00001
"optimizer","SGD"
"optimizer.config.momentum",0.9
"optimizer.config.nesterov",false
"output_activation","softmax"
"python_version","3.9.10"
"run_config.shuffle",true
"shape","rectangle"
"size",131072
"task","AspectTestTask"
"task_version",3
"tensorflow_version","2.8.0"
"test_split",0.2
"test_split_method","shuffled_train_test_split"

{
  "": "AspectTestTask",
  "seed": 1666301679,
  "run_config": {
    "verbose": 0,
  },
  "save_every_epochs": null,
}
    '''
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
            'integer',
            Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit={
            'batch_size': 16,
            'epochs': 3,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.0001
        },
        loss=None,
        early_stopping=None,
        save_every_epochs=-1,
        record_post_training_metrics=True,
        record_times=True,
        )

    worker = Worker(
        None,
        None,
        strategy,
        {},
    )

    results = experiment(worker)
    pprint(marshal.marshal(results), indent=1)

# test_growth_experiment()
# test_simple()

test_from_optimizer()
