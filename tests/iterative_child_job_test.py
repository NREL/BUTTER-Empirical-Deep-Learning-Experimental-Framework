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
        "int64": 1.99699937299130823180304949746497493586e+38,
        "label": "1e",
        "value": "963ccf14-f748-40dd-b2da-638f23041a52"
      },
      "run_id": {
        "type": "UUID",
        "int64": 1.99699937299130823180304949746497493586e+38,
        "label": "1e",
        "value": "963ccf14-f748-40dd-b2da-638f23041a52"
      },
      "context": {
        "cpus": [
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
          17
        ],
        "gpus": [],
        "nodes": [
          0
        ],
        "num_cpus": 10,
        "num_gpus": 0,
        "queue_id": 200,
        "num_nodes": 1,
        "worker_id": {
          "type": "UUID",
          "int64": 3.08412683651886119774123842615453104684e+38,
          "value": "e8061f4c-ea92-4e0a-a4f0-20f860db3a2c"
        },
        "gpu_memory": 0,
        "tensorflow_strategy": "\\\\<class 'tensorflow.python.distribute.distribute_lib._DefaultDistributionStrategy'>"
      },
      "git_hash": "ac1c585",
      "platform": "Linux-3.10.0-1062.9.1.el7.x86_64-x86_64-with-glibc2.17",
      "host_name": "r4i7n35",
      "slurm_job_id": 13231242,
      "python_version": "3.10.8",
      "tensorflow_version": "2.8.1"
    },
    "seed": 1693935724,
    "type": "IterativePruningRunSpec",
    "model_saving": {
      "type": "ModelSavingSpec",
      "save_epochs": [
        0,
        1,
        2,
        4,
        6,
        8,
        10,
        16,
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
      "int64": 1.99699937299130823180304949746497493586e+38,
      "label": "1e",
      "value": "963ccf14-f748-40dd-b2da-638f23041a52"
    },
    "resume_checkpoint": {
      "type": "TrainingExperimentCheckpoint",
      "epoch": {
        "type": "TrainingEpoch",
        "epoch": 30,
        "model_epoch": 30,
        "model_number": 0
      },
      "run_id": {
        "type": "UUID",
        "int64": 1.99699937299130823180304949746497493586e+38,
        "label": "1e",
        "value": "963ccf14-f748-40dd-b2da-638f23041a52"
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
      "epochs": 30,
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
      "cell": "conv_5x5_1x1_valid",
      "stem": "conv_5x5_1x1_same",
      "type": "Lenet",
      "final": {
        "type": "FullyConnectedNetwork",
        "depth": 2,
        "inner": {
          "type": "Dense",
          "units": -1,
          "use_bias": true,
          "activation": "relu",
          "bias_constraint": null,
          "bias_initializer": "Zeros",
          "bias_regularizer": null,
          "kernel_constraint": {
            "class": "ParameterMask"
          },
          "kernel_initializer": "HeUniform",
          "kernel_regularizer": null,
          "activity_regularizer": null
        },
        "input": null,
        "width": 120,
        "output": null,
        "widths": [
          120,
          84
        ],
        "min_width": 84,
        "rectangular": false,
        "flatten_input": true,
        "residual_mode": "none"
      },
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
      },
      "downsample": "max_pool_2x2_2x2_valid",
      "num_stacks": 2,
      "stem_width": 6,
      "cells_per_stack": 1,
      "cell_width_scale_factor": 1,
      "stack_width_scale_factor": 2.6666666666666665,
      "downsample_width_scale_factor": 1
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
        "pruning_rate": 0.5903999999999999
      },
      "new_seed": false,
      "iterations": 6,
      "rewind_epoch": {
        "type": "TrainingEpoch",
        "epoch": 16,
        "model_epoch": 16,
        "model_number": 0
      },
      "rewind_optimizer": true,
      "max_epochs_per_iteration": 30
    },
    "optimizer": {
      "class": "Adam",
      "learning_rate": 0.0012
    },
    "precision": "float32",
    "early_stopping": {
      "class": "EarlyStopping",
      "monitor": "val_loss",
      "patience": 30,
      "min_delta": 0,
      "restore_best_weights": true
    }
  }
}
"""
    deserialized = simplejson.loads(job_command)
    run = marshaling.marshal.demarshal(deserialized)
    experiment_test_util.run_experiment(
        run, True, id=UUID("8e07d977-1f20-4fcb-938f-035138971960")
    )


if __name__ == "__main__":
    iterative_child_job_test()
