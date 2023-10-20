"""
Enqueues jobs from stdin into the JobQueue
"""

import sys


from dmp.model.named.lenet import Lenet
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.training_experiment.run_spec import RunSpec

from dmp.task.run import Run
from dmp.keras_interface.keras_utils import keras_kwcfg

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
    queue_id = 200

    def make_run(
        seed,
    ):
        return Run(
            experiment=TrainingExperiment(
                data={
                    "lth": True,
                    "batch": "lth_mnist_lenet_1",
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
                    "epochs": 24,
                },
                optimizer={
                    "class": "Adam",
                    "learning_rate": 12e-4,
                },
                loss=None,
                early_stopping=keras_kwcfg(
                    "EarlyStopping",
                    monitor="val_loss",
                    min_delta=0,
                    patience=24,
                    restore_best_weights=True,
                ),
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
                    fixed_threshold=-1,
                    exponential_rate=0,
                ),
                resume_checkpoint=None,
            ),
        )

    jobs = []
    seed = int(time.time())
    repetitions = 20
    base_priority = 1000

    for rep in range(repetitions):
        run = make_run(seed + rep)
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
