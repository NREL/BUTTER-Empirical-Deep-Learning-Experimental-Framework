from dmp.context import Context
from dmp.marshaling import marshal
from pprint import pprint
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.model.dense_by_size import DenseBySize
from dmp.layer.dense import Dense
from dmp.dataset.dataset_spec import DatasetSpec
import pytest
import dmp.jobqueue_interface.worker
import tensorflow
import sys

from jobqueue.job import Job
import numpy
import pandas

# from dmp import jobqueue_interface
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)
from dmp.task.run import Run
from dmp.worker import Worker

sys.path.insert(0, "./")


# strategy = dmp.jobqueue_interface.worker.make_strategy(None, [0], 1024*12)
strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
worker = Worker(
    None,
    None,
    None,
    strategy,
    {},
)  # type: ignore


def run_experiment(run):
    context = Context(worker, Job(), run)
    run(context)
    # print("experiment_attrs\n")
    # pprint(results.experiment_attrs)
    # print("experiment_tags\n")
    # pprint(results.experiment_tags)
    # print("run_data\n", results.run_data)
    # print("run_history\n", results.run_history)
    # print("run_extended_history\n", results.run_extended_history)
    # return results


def test_simple():
    run = Run(
        experiment=TrainingExperiment(
            data={"test": True},
            precision="float32",
            dataset=DatasetSpec(
                # 'titanic',
                # 'pmlb',
                "GaussianClassificationDataset_2_10_100",
                # # 'GaussianRegressionDataset_20_100',
                "synthetic",
                "shuffled_train_test_split",
                0.2,
                0.05,
                0.0,
            ),
            model=DenseBySize(
                input=None,
                output=None,
                shape="rectangle",
                size=16384,
                depth=4,
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
                "batch_size": 256,
                "epochs": 5,
            },
            optimizer={
                "class": "Adam",
                "learning_rate": 0.001,
            },
            loss=None,
            early_stopping=None,
        ),
        run=RunSpec(
            seed=0,
            data={
                "test": True,
            },
            record_post_training_metrics=True,
            record_times=True,
            model_saving=None,
            resume_checkpoint=None,
        ),
    )

    run_experiment(run)


if __name__ == "__main__":
    test_simple()
