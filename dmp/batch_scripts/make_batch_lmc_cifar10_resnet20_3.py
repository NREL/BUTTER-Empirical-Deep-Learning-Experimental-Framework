"""
Enqueues jobs from stdin into the JobQueue
"""

import math
import sys

import numpy


from dmp.model.named.lenet import Lenet
from dmp.model.named.resenet20 import Resnet20
from dmp.task.experiment.lth.lth_experiment import LTHExperiment
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch

from dmp.task.run import Run
from dmp.keras_interface.keras_utils import keras_kwcfg
from tests.paper_param import get_paper_param

sys.path.insert(0, "./")

from dmp.dataset.dataset_spec import DatasetSpec

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

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
        param_name,
        param,
        pruning_configs,
    ):
        return Run(
            experiment=LTHExperiment(
                data={
                    "lmc": True,
                    "batch": f"lmc_cifar10_resnet20_{param_name}_3",
                    "group": f"lmc_cifar10_resnet20_{param_name}",
                    "supergroup": "lmc_cifar10_resnet20",
                    "model_family": "resnet",
                    "model_name": "Resnet20",
                    "resnet_depth": 20,
                },
                precision="float32",
                dataset=DatasetSpec(
                    "cifar10",
                    "keras",
                    "shuffled_train_test_split",
                    0.2,
                    0.05,
                    0.0,
                ),
                model=Resnet20(),
                fit={
                    "batch_size": param["batch"],
                    "epochs": int(
                        math.ceil(param["batch"] * param["train_step"] / 60000)
                    ),  # 60000 is the number of training images in CIFAR10
                },
                optimizer={
                    "class": param["optimizer"],
                    "learning_rate": param["learning_rate"],
                },
                loss=None,
                early_stopping=keras_kwcfg(
                    "DMPEarlyStopping",
                    monitor="val_loss",
                    min_delta=0,
                    patience=50,
                    restore_best_weights=True,
                ),
                pruning_configs=pruning_configs,
                num_additional_seeds_per_config=1,
            ),
            run=RunSpec(
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
                    exponential_rate=math.pow(2, 1 / 4.0),
                ),
                saved_models=[],
                resume_checkpoint=None,
            ),
        )

    jobs = []
    seed = int(time.time())
    repetitions = 10
    base_priority = 2000

    for param_name in ["standard", "low"]:
        # [.8^(2) = .64 (36%), .8 (20%), .8^(1/2)~=.894 (10.6%), .8^(1/4) ~= .945 (5.4%)] pruning per IMP iteration
        #         to target of <3.5% LeNet (16 iters), 3.5% ResNet (16 iters), 0.6% (24 iters) VGG:
        param = get_paper_param("Linear_Mode_Connectivity", "RESNET", param_name)

        pruning_target = 0.005
        base_pruning_rate = param["pruning_rate"]
        base_survival_rate = 1.0 - base_pruning_rate

        pruning_configs = []
        for survival_rate in [
            base_survival_rate**4,
            base_survival_rate**2,
            base_survival_rate,
            base_survival_rate ** (1 / 2),
            base_survival_rate ** (1 / 4),
        ]:
            pruning_iterations = int(
                numpy.ceil(numpy.log(pruning_target) / numpy.log(survival_rate))
            )
            pruning_rate = 1.0 - survival_rate

            for rewind_epoch in [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                12,
                16,
            ]:
                pruning_configs.append(
                    PruningConfig(
                        iterations=pruning_iterations,
                        method=MagnitudePruner(pruning_rate),
                        max_epochs_per_iteration=int(
                            math.ceil(param["batch"] * param["train_step"] / 60000)
                        ),
                        rewind_epoch=TrainingEpoch(
                            epoch=rewind_epoch,
                            fit_number=0,
                            fit_epoch=rewind_epoch,
                        ),
                        rewind_optimizer=True,
                        new_seed=False,
                    )
                )

        for rep in range(repetitions):
            run = make_run(
                seed + len(jobs),
                param_name,
                param,
                pruning_configs,
            )
            jobs.append(
                Job(
                    priority=base_priority + len(jobs),
                    command=marshal.marshal(run),
                )
            )

    print(f"Generated {len(jobs)} jobs.")
    # pprint(jobs)
    credentials = jobqueue.load_credentials("dmp")
    job_queue = JobQueue(credentials, queue_id, check_table=False)
    job_queue.push(jobs)
    print(f"Enqueued {len(jobs)} jobs.")


if __name__ == "__main__":
    main()
