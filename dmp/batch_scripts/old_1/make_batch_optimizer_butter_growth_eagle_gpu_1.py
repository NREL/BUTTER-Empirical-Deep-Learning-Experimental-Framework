"""
Enqueues jobs from stdin into the JobQueue
"""

import sys

import numpy
import pandas
from tensorflow.python.framework.ops import re

from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.dense_by_size import DenseBySize
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
    queue_id = 12

    def make_experiment(
        seed,
        dataset,
        size,
        depth,
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
                "optimizer_butter_growth_1": True,
                "butter_growth": True,
            },
            run_tags={
                "make_batch_optimizer_butter_growth_eagle_gpu_1": True,
            },
            batch="make_batch_optimizer_butter_growth_eagle_gpu_1",
            precision="float32",
            dataset=DatasetSpec(
                dataset,
                "pmlb",
                "shuffled_train_test_split",
                0.2,
                0.05,
                0.0,
            ),
            model=DenseBySize(
                input=None,
                output=None,
                shape="rectangle",
                size=size,
                depth=depth,
                search_method="integer",
                inner=Dense.make(
                    -1,
                    {
                        "activation": "relu",
                        "kernel_initializer": "GlorotUniform",
                    },
                ),
            ),
            fit={
                "batch_size": batch_size,
                "epochs": 3000 * 100,
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
                verbose=0,
                mode="min",
                baseline=None,
                # start_from_epoch=0,
            ),
            scaling_method=WidthScaler(),
            transfer_method=OverlayTransfer(),
            growth_scale=growth_scale,
            initial_size=32,
            max_epochs_per_stage=3000,
            max_equivalent_epoch_budget=6000,
        )

    sweep_config = list(
        {
            # 'dataset': ['mnist'],
            "dataset": [
                "201_pol",
                "529_pollen",
                "537_houses",
                "connect_4",
                "sleep",
                "wine_quality_white",
                "adult",
                "nursery",
                "splice",
                "294_satellite_image",
                "banana",
                "505_tecator",
                "poker",
            ],
            # ['201_pol', '529_pollen', '537_houses',  'connect_4', 'mnist', 'sleep', 'wine_quality_white', 'adult', 'nursery', 'splice', '294_satellite_image', 'banana', '505_tecator', 'poker'],
            "size": [16777216],
            "depth": [2, 3, 4, 5, 6],
            "batch_size": [32, 64, 128, 256],
            "optimizer": ["Adam", "SGD", "RMSprop", "Adagrad"],
            "learning_rate": [1e-2, 1e-3, 1e-4, 1e-5],
            "momentum": [0.0, 0.9],
            "min_delta": [0.1, 0.01, 0.001],
            "growth_scale": [2.0],
        }.items()
    )

    jobs = []
    seed = int(time.time())
    repetitions = 1
    base_priority = 2000000

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
