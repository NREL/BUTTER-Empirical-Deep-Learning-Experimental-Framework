"""
Enqueues jobs from stdin into the JobQueue
"""

import math
import sys

import numpy
from dmp.layer.dense import Dense
from dmp.model.dense_by_size import DenseBySize


from dmp.model.named.lenet import Lenet
from dmp.task.experiment.lth.lth_experiment import LTHExperiment
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

from dmp.task.run import Run
from dmp.keras_interface.keras_utils import keras_kwcfg

sys.path.insert(0, "./")

from dmp.dataset.dataset_spec import DatasetSpec


from dmp.marshaling import marshal

import time

import jobqueue
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue

import sys


def main():
    queue_id = 10

    def make_run(
        seed,
        dataset,
        size,
        depth,
        batch_size,
        optimizer,
        node_config,
    ):
        return Run(
            experiment=TrainingExperiment(
                data={
                    "butter_e": True,
                    "butter_e_optimizer": True,
                    "batch": f"optimizer_energy_batch_1_{node_config}",
                    "node_config": node_config,
                },
                precision="float32",
                dataset=DatasetSpec(
                    dataset,
                    "pmlb",
                    "shuffled_train_test_split",
                    0.15,
                    0.05,
                    0.0,
                ),
                model=DenseBySize(
                    input=None,
                    output=None,
                    shape="rectangle",
                    size=size,
                    depth=depth,
                    search_method="float",
                    inner=Dense.make(
                        -1,
                        {
                            "activation": "relu",
                            "kernel_initializer": "HeUniform",
                        },
                    ),
                ),
                fit={
                    "batch_size": batch_size,
                    "epochs": 3000,
                },
                optimizer=optimizer,
                loss=None,
                early_stopping=keras_kwcfg(
                    "DMPEarlyStopping",
                    monitor="val_loss",
                    min_delta=0,
                    patience=50,
                    restore_best_weights=True,
                ),
            ),
            run=RunSpec(
                seed=seed,
                data={},
                record_post_training_metrics=False,
                record_times=True,
                model_saving=None,
                saved_models=[],
                resume_checkpoint=None,
            ),
        )

    jobs = []
    seed = int(time.time())
    repetitions = 10
    base_priority = 1000

    cpu_jobs = []
    gpu_jobs = []
    for node_config, jobs in [
        ("eagle_cpu", cpu_jobs),
        ("eagle_gpu", gpu_jobs),
    ]:
        for rep in range(repetitions):
            for dataset in [
                "201_pol",
                "529_pollen",
                "537_houses",
                "connect_4",
                "mnist",
                "sleep",
                "wine_quality_white",
                "adult",
                "nursery",
                "splice",
                "294_satellite_image",
                "banana",
                "505_tecator",
            ]:
                for optimizer in [
                    {"class": "Adam", "learning_rate": 0.01},
                    {"class": "Adam", "learning_rate": 0.001},
                    {"class": "Adam", "learning_rate": 0.0001},
                    {"class": "Adam", "learning_rate": 0.00001},
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.01,
                    #         "momentum": 0.0,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.001,
                    #         "momentum": 0.0,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.0001,
                    #         "momentum": 0.0,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.00001,
                    #         "momentum": 0.0,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.01,
                    #         "momentum": 0.9,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.001,
                    #         "momentum": 0.9,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.0001,
                    #         "momentum": 0.9,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "SGD",
                    #     "config": {
                    #         "learning_rate": 0.00001,
                    #         "momentum": 0.9,
                    #         "nesterov": False,
                    #     },
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.01, "momentum": 0.0},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.001, "momentum": 0.0},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.0001, "momentum": 0.0},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.00001, "momentum": 0.0},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.01, "momentum": 0.9},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.001, "momentum": 0.9},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.0001, "momentum": 0.9},
                    # },
                    # {
                    #     "class_name": "RMSprop",
                    #     "config": {"learning_rate": 0.00001, "momentum": 0.9},
                    # },
                    # {"class_name": "Adagrad", "config": {"learning_rate": 0.01}},
                    # {"class_name": "Adagrad", "config": {"learning_rate": 0.001}},
                    # {"class_name": "Adagrad", "config": {"learning_rate": 0.0001}},
                    # {"class_name": "Adagrad", "config": {"learning_rate": 0.00001}},
                    # {"class_name": "Adadelta", "config": {"learning_rate": 0.01}},
                    # {"class_name": "Adadelta", "config": {"learning_rate": 0.001}},
                    # {"class_name": "Adadelta", "config": {"learning_rate": 0.0001}},
                    # {"class_name": "Adadelta", "config": {"learning_rate": 0.00001}},
                    # {"class_name": "Adamax", "config": {"learning_rate": 0.01}},
                    # {"class_name": "Adamax", "config": {"learning_rate": 0.001}},
                    # {"class_name": "Adamax", "config": {"learning_rate": 0.0001}},
                    # {"class_name": "Adamax", "config": {"learning_rate": 0.00001}},
                    # {"class_name": "Nadam", "config": {"learning_rate": 0.01}},
                    # {"class_name": "Nadam", "config": {"learning_rate": 0.001}},
                    # {"class_name": "Nadam", "config": {"learning_rate": 0.0001}},
                    # {"class_name": "Nadam", "config": {"learning_rate": 0.00001}},
                ]:
                    for depth in [
                        2,
                        3,
                        4,
                        5,
                        6,
                    ]:
                        for batch_size in [
                            16,
                            32,
                            64,
                            128,
                            256,
                        ]:
                            for size in [
                                # 32,
                                64,
                                128,
                                256,
                                512,
                                1024,
                                2048,
                                4096,
                                8192,
                                16384,
                                32768,
                                65536,
                                131072,
                                262144,
                                524288,
                                1048576,
                                2097152,
                                4194304,
                                8388608,
                                16777216,
                            ]:
                                run = make_run(
                                    seed + len(jobs),
                                    dataset,
                                    size,
                                    depth,
                                    batch_size,
                                    optimizer,
                                    node_config,
                                )
                                jobs.append(
                                    Job(
                                        priority=base_priority + len(jobs),
                                        command=marshal.marshal(run),
                                    )
                                )

    credentials = jobqueue.load_credentials("dmp")
    for queue_id, jobs in [
        (51, cpu_jobs),
        (52, gpu_jobs),
    ]:
        print(f"Generated {len(jobs)} jobs.")
        job_queue = JobQueue(credentials, queue_id, check_table=False)
        job_queue.push(jobs)
        print(f"Enqueued {len(jobs)} jobs in queue {queue_id}.")


if __name__ == "__main__":
    main()
