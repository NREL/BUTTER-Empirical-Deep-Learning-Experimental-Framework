"""
Enqueues jobs from stdin into the JobQueue
"""

import math
import sys
from uuid import uuid4
from jobqueue.connect import load_credentials

import numpy
from dmp.batch_scripts.batch_util import enqueue_batch_of_runs


from dmp.model.named.lenet import Lenet
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.run_entry import RunEntry
from dmp.task.experiment.lth.lth_experiment import LTHExperiment
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)
from dmp.task.experiment.pruning_experiment.pruning_method.random_pruner import (
    RandomPruner,
)
from dmp.task.experiment.training_experiment.run_spec import RunConfig
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

from dmp.task.run import Run
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.task.run_status import RunStatus

sys.path.insert(0, "./")

from dmp.dataset.dataset_spec import DatasetSpec


from dmp.marshaling import marshal

import time

import sys


def main():
    def make_run(
        seed,
        pruning_configs,
    ):
        return Run(
            experiment=LTHExperiment(
                data={
                    "lmc": True,
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
                    "batch_size": 60,
                    "epochs": 256,
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
                    patience=3,
                    restore_best_weights=True,
                ),
                pruning_configs=pruning_configs,
                num_additional_seeds_per_config=1,
            ),
            config=RunConfig(
                seed=seed,
                data={
                    "batch": "lmc_mnist_lenet_4",
                },
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

    runs = []
    seed = int(time.time())
    repetitions = 10
    base_priority = 1000

    # [.8^(2) = .64 (36%), .8 (20%), .8^(1/2)~=.894 (10.6%), .8^(1/4) ~= .945 (5.4%)] pruning per IMP iteration
    #         to target of <3.5% LeNet (16 iters), 3.5% ResNet (16 iters), 0.6% (24 iters) VGG:

    pruning_target = 0.01
    pruning_configs = []
    for survival_rate in [
        0.8**4,
        0.8**2,
        0.8,
        0.8 ** (1 / 2),
        0.8 ** (1 / 4),
        0.8 ** (1 / 8),
        0.8 ** (1 / 16),
        0.8 ** (1 / 32),
    ]:
        pruning_iterations = int(
            numpy.ceil(numpy.log(pruning_target) / numpy.log(survival_rate))
        )
        pruning_rate = 1.0 - survival_rate

        for rewind_epoch in [
            0,
            1,
            2,
            # 3,
            4,
            # 6,
            8,
            # 12,
            # 16,
            # 24,
        ]:
            pruning_configs.extend(
                [
                    PruningConfig(
                        iterations=pruning_iterations,
                        method=MagnitudePruner(pruning_rate),
                        max_epochs_per_iteration=128,
                        rewind_epoch=TrainingEpoch(
                            epoch=rewind_epoch,
                            fit_number=0,
                            fit_epoch=rewind_epoch,
                        ),
                        rewind_optimizer=True,
                        new_seed=False,
                    ),
                    PruningConfig(
                        iterations=pruning_iterations,
                        method=RandomPruner(pruning_rate),
                        max_epochs_per_iteration=128,
                        rewind_epoch=TrainingEpoch(
                            epoch=rewind_epoch,
                            fit_number=0,
                            fit_epoch=rewind_epoch,
                        ),
                        rewind_optimizer=True,
                        new_seed=False,
                    ),
                ]
            )

    runs = [
        make_run(
            seed + i,
            pruning_configs,
        )
        for i in range(repetitions)
    ]

    enqueue_batch_of_runs(runs, 11, 0)


if __name__ == "__main__":
    main()
