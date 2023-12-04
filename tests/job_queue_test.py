from uuid import UUID

from jobqueue.job_queue import JobQueue
from dmp.context import Context
from dmp.marshaling import marshal
from pprint import pprint
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.training_experiment.run_spec import RunConfig
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
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.structure.batch_norm_block import BatchNormBlock
from dmp.structure.sequential_model import SequentialModel
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)
from dmp.task.run import Run
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.marshaling import marshal

import tests.experiment_test_util as experiment_test_util

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

save_id = UUID("a69b6248-9790-4641-9620-0942fd20a442")


def queue_jobs():
    queue_id = 100
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
                "epochs": 10,
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
                patience=100,
                restore_best_weights=True,
            ),
        ),
        config=RunConfig(
            seed=1,
            data={
                "test": True,
                "type": "my special run",
            },
            record_post_training_metrics=False,
            record_times=True,
            model_saving=ModelSavingSpec(
                save_initial_model=False,
                save_trained_model=True,
                save_fit_epochs=[],
                save_epochs=[
                    1,
                    2,
                    3,
                ],
                fixed_interval=0,
                fixed_threshold=0,
                exponential_rate=0,
            ),
            resume_checkpoint=None,
        ),
    )

    job = Job(
        id=save_id,
        priority=0,
        command=marshal.marshal(run),
    )

    credentials = load_credentials("dmp")
    job_queue = JobQueue(credentials, queue_id, check_table=False)
    job_queue.push([job])


def run_jobs():
    queue_id = 100

    credentials = load_credentials("dmp")
    job_queue = JobQueue(credentials, queue_id, check_table=False)
    job = job_queue.pop()
    if job is None:
        print("No job found!")
        return

    worker = Worker(
        None,  # type: ignore
        PostgresInterface(credentials),  # type: ignore
        dmp.jobqueue_interface.worker.make_strategy(None, None, None),  # type: ignore
        {},  # type: ignore
    )  # type: ignore
    run: Run = marshal.demarshal(job.command)
    context = Context(worker, job, run)
    run(context)


if __name__ == "__main__":
    # queue_jobs()
    run_jobs()
