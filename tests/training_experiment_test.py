import sys

import tensorflow

from dmp.worker import Worker
sys.path.insert(0, './')
import pytest

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.layer.dense import Dense
from dmp.model.dense_by_size import DenseBySize

from dmp.task.training_experiment.training_experiment import TrainingExperiment

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
            'rectangle',
            16384,
            3,
            Dense({
                'activation' : 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit_config={
            'batch_size':16,
            'epochs': 10,
        },
        optimizer={'type': 'adam', 'config': {'learning_rate': 0.0001},},
        loss=None,
        early_stopping=None,
        save_every_epochs=-1
        )

    worker = Worker(
        None,
        None,
        tensorflow.distribute.get_strategy(),
        {
        },
    )
    
    results = experiment(worker)
    print(results)
    