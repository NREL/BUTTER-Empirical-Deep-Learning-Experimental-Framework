from uuid import UUID

import simplejson
from dmp import marshaling
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


def iterative_child_job_test():
    job_command = """
{
  "run": {
    "data": {
      "job_id": {
        "type": "UUID",
        "label": "1a",
        "value": "001a8336-4303-48e9-b88a-29dbea4eaf1a"
      },
      "run_id": {
        "type": "UUID",
        "label": "1a",
        "value": "001a8336-4303-48e9-b88a-29dbea4eaf1a"
      },
      "context": {
        "cpus": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ],
        "gpus": [],
        "nodes": [
          0
        ],
        "num_cpus": 8,
        "num_gpus": 0,
        "queue_id": 200,
        "num_nodes": 1,
        "worker_id": {
          "type": "UUID",
          "value": "1704f3e2-f9fd-490d-af0e-8324e635cd15"
        },
        "gpu_memory": 0,
        "tensorflow_strategy": "<class 'tensorflow.python.distribute.distribute_lib._DefaultDistributionStrategy'>"
      },
      "git_hash": "921c228",
      "platform": "Linux-3.10.0-1062.9.1.el7.x86_64-x86_64-with-glibc2.17",
      "host_name": "r4i6n1",
      "slurm_job_id": 13306753,
      "python_version": "3.10.8",
      "tensorflow_version": "2.8.1"
    },
    "seed": 1694811813,
    "type": "IterativePruningRunSpec",
    "model_saving": {
      "type": "ModelSavingSpec",
      "save_epochs": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        16,
        24,
        32
      ],
      "fixed_interval": 1,
      "fixed_threshold": -1,
      "exponential_rate": 0,
      "save_model_epochs": [],
      "save_initial_model": true,
      "save_trained_model": true
    },
    "record_times": true,
    "rewind_run_id": {
      "type": "UUID",
      "value": "35d90ef7-29c0-4c7c-9ff3-a54a30e8dfda"
    },
    "resume_checkpoint": {
      "type": "TrainingExperimentCheckpoint",
      "epoch": {
        "type": "TrainingEpoch",
        "epoch": 704,
        "model_epoch": 32,
        "model_number": 21
      },
      "run_id": {
        "type": "UUID",
        "label": "1a",
        "value": "001a8336-4303-48e9-b88a-29dbea4eaf1a"
      },
      "load_mask": true,
      "load_optimizer": true
    },
    "prune_first_iteration": true,
    "record_post_training_metrics": true
  },
  "type": "Run",
  "experiment": {
    "fit": {
      "epochs": 32,
      "batch_size": 60
    },
    "data": {
      "lth": true,
      "batch": "lth_mnist_lenet_1",
      "ml_task": "classification",
      "model_name": "lenet_relu",
      "input_shape": [
        28,
        28,
        1
      ],
      "model_family": "lenet",
      "output_shape": [
        10
      ],
      "data_set_size": 70000,
      "test_set_size": 10000,
      "train_set_size": 56500,
      "network_description": {},
      "num_free_parameters": 61706,
      "validation_set_size": 3500
    },
    "loss": {
      "class": "CategoricalCrossentropy"
    },
    "type": "IterativePruningExperiment",
    "model": {
      "type": "Lenet",
      "input": {
        "name": "dmp_8",
        "type": "Input",
        "shape": [
          28,
          28,
          1
        ]
      },
      "output": {
        "type": "Dense",
        "units": 10,
        "use_bias": true,
        "activation": "softmax",
        "bias_constraint": null,
        "bias_initializer": "Zeros",
        "bias_regularizer": null,
        "kernel_constraint": null,
        "kernel_initializer": {
          "class": "GlorotUniform"
        },
        "kernel_regularizer": null,
        "activity_regularizer": null
      }
    },
    "dataset": {
      "name": "mnist",
      "type": "DatasetSpec",
      "method": "shuffled_train_test_split",
      "source": "keras",
      "test_split": 0.14285714285714285,
      "label_noise": 0,
      "validation_split": 0.05
    },
    "pruning": {
      "type": "PruningConfig",
      "method": {
        "type": "MagnitudePruner",
        "pruning_rate": 0.19999999999999996
      },
      "new_seed": false,
      "iterations": 30,
      "rewind_epoch": {
        "type": "TrainingEpoch",
        "epoch": 1,
        "model_epoch": 1,
        "model_number": 0
      },
      "rewind_optimizer": true,
      "max_epochs_per_iteration": 32
    },
    "optimizer": {
      "class": "Adam",
      "learning_rate": 0.0012
    },
    "precision": "float32",
    "early_stopping": {
      "class": "EarlyStopping",
      "monitor": "val_loss",
      "patience": 32,
      "min_delta": 0,
      "restore_best_weights": true
    }
  }
}
"""
    deserialized = simplejson.loads(job_command)
    run = marshaling.marshal.demarshal(deserialized)
    experiment_test_util.run_experiment(
        run, True, id=UUID("001a8336-4303-48e9-b88a-29dbea4eaf1a")
    )


if __name__ == "__main__":
    iterative_child_job_test()
