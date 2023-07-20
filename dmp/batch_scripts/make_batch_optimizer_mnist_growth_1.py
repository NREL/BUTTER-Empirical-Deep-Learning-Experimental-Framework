"""
Enqueues jobs from stdin into the JobQueue
"""

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
from dmp.keras_interface.keras_utils import keras_kwcfg
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
        width,
        batch_size,
        optimizer,
        learning_rate,
        momentum,
        growth_scale,
        min_delta,
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

        return GrowthExperiment(
            seed=seed,
            experiment_tags={
                "mnist_cnn": True,
                "mnist_simple_cnn_growth_v1": True,
            },
            run_tags={},
            batch="optimizer_cnn_mnist_growth_1",
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
                num_stacks=3,
                cells_per_stack=1,
                stem="conv_5x5_1x1_same",
                downsample="max_pool_2x2_2x2_valid",
                cell="conv_5x5_1x1_same",
                final=FullyConnectedNetwork(
                    input=None,
                    output=None,
                    widths=[width * 2, width * 2],
                    residual_mode="none",
                    flatten_input=True,
                    inner=Dense.make(-1, {}),
                ),
                stem_width=width,
                stack_width_scale_factor=1.0,
                downsample_width_scale_factor=1.0,
                cell_width_scale_factor=1.0,
            ),
            fit={
                "batch_size": batch_size,
                "epochs": 1024 * 128,
            },
            optimizer=optimizer,
            loss=None,
            early_stopping=keras_kwcfg(
                "EarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=16,
                restore_best_weights=True,
            ),
            record=RunSpec(
                post_training_metrics=True,
                times=True,
                model=None,
                metrics=None,
            ),
            growth_trigger=keras_kwcfg(
                "ProportionalStopping",
                restore_best_weights=True,
                monitor="val_loss",
                min_delta=min_delta,
                patience=0,
                verbose=1,
                mode="min",
                baseline=None,
                # start_from_epoch=0,
            ),
            scaling_method=WidthScaler(),
            transfer_method=OverlayTransfer(),
            growth_scale=growth_scale,
            initial_size=16,
            max_epochs_per_stage=1024,
            max_equivalent_epoch_budget=2048,
        )

    sweep_config = list(
        {
            "width": [2, 3, 4, 5, 6, 7, 8],
            "batch_size": [8, 16, 32, 64, 128, 256, 512],
            "optimizer": ["Adam", "SGD", "RMSprop", "Adagrad"],
            "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
            "momentum": [0.0, 0.9],
            "min_delta": [0.1, 0.01, 0.001],
            "growth_scale": [
                2.0,
            ],
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
