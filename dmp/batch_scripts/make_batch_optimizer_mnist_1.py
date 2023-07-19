"""
Enqueues jobs from stdin into the JobQueue
"""

import math
import sys

import numpy
import pandas
from tensorflow.python.framework.ops import re

from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)
from dmp.task.experiment.run_spec import (
    RunSpec,
)
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

sys.path.insert(0, "./")

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.layer.dense import Dense

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from pprint import pprint

from dmp.marshaling import marshal

import time

import jobqueue
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue
import numpy

from command_line_tools import command_line_config

import sys


def main():
    queue_id = 11

    def make_experiment(
        seed,
        fc_scale,
        stem_width,
        batch_size,
        optimizer,
        learning_rate,
        momentum,
    ):
        if optimizer == "Adam" and momentum != 0.0:
            return None

        optimizer = {
            "class": optimizer,
            "learning_rate": learning_rate,
            "momentum": momentum,
        }

        if optimizer == "Adam":
            del optimizer["momentum"]

        tags = {
            "mnist_cnn": True,
            "mnist_simple_cnn_v1": True,
            "model_family": "lenet",
            "model_genus": "lenet_relu",
            "model_fc_scale": fc_scale,
        }

        if fc_scale == 1.0 and stem_width == 6:
            tags["model_species"] = "lenet"

        return TrainingExperiment(
            seed=seed,
            batch="optimizer_cnn_mnist_1",
            experiment_tags=tags,
            run_tags={},
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
                    widths=numpy.round(numpy.array([120, 84]) * fc_scale)
                    .astype(int)
                    .tolist(),
                    residual_mode="none",
                    flatten_input=True,
                    inner=Dense.make(-1, {}),
                ),
                stem_width=stem_width,
                stack_width_scale_factor=16.0 / 6.0,
                downsample_width_scale_factor=1.0,
                cell_width_scale_factor=1.0,
            ),
            fit={
                "batch_size": batch_size,
                "epochs": 1024,
            },
            optimizer=optimizer,
            loss=None,
            early_stopping=make_keras_kwcfg(
                "EarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=32,
                restore_best_weights=True,
            ),
            record=RunSpec(
                post_training_metrics=True,
                times=True,
                model=None,
                metrics=None,
            ),
        )

    sweep_config = list(
        {
            "fc_scale": [
                1.0,
            ],
            "stem_width": [3, 4, 5, 6, 7, 8],
            "batch_size": [
                8,
                16,
                32,
                64,
                128,
                256,
            ],
            "optimizer": [
                "Adam",
            ],  #'SGD', 'RMSprop', 'Adagrad'
            "learning_rate": [3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3],
            "momentum": [0.0, 0.9],
        }.items()
    )

    jobs = []
    seed = int(time.time())
    repetitions = 10
    base_priority = 0

    def do_sweep(i, config):
        if i < 0:
            for rep in range(repetitions):
                experiment = make_experiment(seed + len(jobs), **config)
                if experiment is not None:
                    jobs.append(
                        Job(
                            priority=base_priority + len(jobs),
                            command=marshal.marshal(experiment),
                        )
                    )
        else:
            key, values = sweep_config[i]
            for v in values:
                config[key] = v
                do_sweep(i - 1, config)

    do_sweep(len(sweep_config) - 1, {})

    print(f"Generated {len(jobs)} jobs.")
    # pprint(jobs)
    credentials = jobqueue.load_credentials("dmp")
    job_queue = JobQueue(credentials, queue_id, check_table=False)
    job_queue.push(jobs)
    print(f"Enqueued {len(jobs)} jobs.")


if __name__ == "__main__":
    main()
