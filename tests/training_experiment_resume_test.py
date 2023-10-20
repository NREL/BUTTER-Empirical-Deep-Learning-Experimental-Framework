from uuid import UUID
from dmp.marshaling import marshal
from pprint import pprint
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

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

save_id = UUID("a69b6248-9790-4641-9620-0942fd20a442")
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
                "epochs": 6,
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
                patience=10,
                restore_best_weights=True,
            ),
        ),
        run=RunSpec(
            seed=seed,
            data={
                "test": True,
            },
            record_post_training_metrics=False,
            record_times=True,
            model_saving=ModelSavingSpec(
                save_initial_model=False,
                save_trained_model=True,
                save_fit_epochs=[],
                save_epochs=[
                    3,
                ],
                fixed_interval=0,
                fixed_threshold=0,
                exponential_rate=0,
            ),
            resume_checkpoint=None,
        ),
    )

    experiment_test_util.run_experiment(run, use_database=True, id=save_id)

    run.run.resume_checkpoint = TrainingExperimentCheckpoint(
        run_id=save_id,
        load_mask=True,
        load_optimizer=True,
        epoch=TrainingEpoch(
            epoch=3,
            fit_number=0,
            fit_epoch=3,
        ),
    )

    run.run.model_saving = None

    experiment_test_util.run_experiment(run, use_database=True)


if __name__ == "__main__":
    test_mnist_lenet()
    # test_resume()
