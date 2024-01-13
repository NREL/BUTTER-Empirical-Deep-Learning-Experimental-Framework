import math
from uuid import UUID
import uuid
from dmp.marshaling import marshal
from pprint import pprint
from dmp.model.named.lenet import Lenet
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.training_experiment.run_spec import RunConfig
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

from dmp.model.dense_by_size import DenseBySize
from dmp.layer.dense import Dense
from dmp.dataset.dataset_spec import DatasetSpec
import pytest
import dmp.script.worker
import tensorflow
import sys
from jobqueue.connect import load_credentials

from jobqueue.job import Job
import numpy
import pandas
from tensorflow.python.framework.ops import re
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense_conv import DenseConv

# from dmp import script
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.layer_factory_model import LayerFactoryModel
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.structure.batch_norm_block import BatchNormBlock
from dmp.structure.sequential_model import SequentialModel
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)
from dmp.task.run import Run
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import keras_kwcfg


import tests.experiment_test_util as experiment_test_util

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

# save_id = UUID("a69b6248-9790-4641-9620-0942fd20a442")
save_id = uuid.uuid4()

seed = 10


def test_mnist_lenet():
    # save_id = UUID("a69b6248-9790-4641-9620-0942fd20a442")
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
                10 / 70.0,
                0.05,
                0.0,
            ),
            model=Lenet(),
            fit={
                "batch_size": 128,
                "epochs": 6,
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
        ),
        config=RunConfig(
            seed=seed,
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

    experiment_test_util.run_experiment(run, use_database=True, id=save_id)

    run.config.resume_checkpoint = TrainingExperimentCheckpoint(
        run_id=save_id,
        load_mask=True,
        load_optimizer=True,
        epoch=TrainingEpoch(
            epoch=3,
            fit_number=0,
            fit_epoch=3,
        ),
    )

    run.config.model_saving = None

    experiment_test_util.run_experiment(run, use_database=True, id=save_id)


if __name__ == "__main__":
    test_mnist_lenet()
    # test_resume()
