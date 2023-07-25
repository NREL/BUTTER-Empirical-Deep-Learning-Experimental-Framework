from uuid import UUID
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


def test_mnist_lenet():
    save_id = UUID("a69b6248-9790-4641-9620-0942fd20a442")
    run = Run(
        experiment=TrainingExperiment(
            data={
                "test": True,
                "model_family": "lenet",
                "model_name": "lenet_relu",
            },
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
                    widths=[120, 84],
                    residual_mode="none",
                    flatten_input=True,
                    inner=Dense.make(-1, {}),
                ),
                stem_width=6,
                stack_width_scale_factor=16.0 / 6.0,
                downsample_width_scale_factor=1.0,
                cell_width_scale_factor=1.0,
            ),
            fit={
                "batch_size": 128,
                "epochs": 6,
            },
            optimizer={
                "class": "Adam",
                "learning_rate": 0.01,
            },
            loss=None,
            early_stopping=keras_kwcfg(
                "EarlyStopping",
                monitor="val_loss",
                min_delta=0,
                patience=10,
                restore_best_weights=True,
            ),
        ),
        run=RunSpec(
            seed=1,
            data={
                "test": True,
            },
            record_post_training_metrics=False,
            record_times=True,
            model_saving=ModelSavingSpec(
                save_initial_model=False,
                save_trained_model=True,
                save_model_epochs=[],
                save_epochs=[
                    3,
                ],
                fixed_interval=0,
                fixed_threshold=0,
                exponential_rate=0,
            ),
            resume_checkpoint=None,
        ),
    )

    experiment_test_util.run_experiment(run, use_database=True, id=save_id)

    run.run.resume_checkpoint = TrainingExperimentCheckpoint(
        run_id=save_id,
        load_mask=True,
        load_optimizer=True,
        epoch=TrainingEpoch(
            epoch=3,
            model_number=0,
            model_epoch=3,
        ),
    )

    run.run.model_saving = None

    experiment_test_util.run_experiment(run, use_database=True)


# def test_mnist_lenet():
#     run = Run(
#         experiment=TrainingExperiment(
#             data={
#                 "test": True,
#                 "model_family": "lenet",
#                 "model_name": "lenet_relu",
#             },
#             precision="float32",
#             dataset=DatasetSpec(
#                 "mnist",
#                 "keras",
#                 "shuffled_train_test_split",
#                 0.2,
#                 0.05,
#                 0.0,
#             ),
#             model=CNNStack(
#                 input=None,
#                 output=None,
#                 num_stacks=2,
#                 cells_per_stack=1,
#                 stem="conv_5x5_1x1_same",
#                 downsample="max_pool_2x2_2x2_valid",
#                 cell="conv_5x5_1x1_valid",
#                 final=FullyConnectedNetwork(
#                     input=None,
#                     output=None,
#                     widths=[120, 84],
#                     residual_mode="none",
#                     flatten_input=True,
#                     inner=Dense.make(-1, {}),
#                 ),
#                 stem_width=6,
#                 stack_width_scale_factor=16.0 / 6.0,
#                 downsample_width_scale_factor=1.0,
#                 cell_width_scale_factor=1.0,
#             ),
#             fit={
#                 "batch_size": 128,
#                 "epochs": 15,
#             },
#             optimizer={
#                 "class": "Adam",
#                 "learning_rate": 0.01,
#             },
#             loss=None,
#             early_stopping=keras_kwcfg(
#                 "EarlyStopping",
#                 monitor="val_loss",
#                 min_delta=0,
#                 patience=4,
#                 restore_best_weights=True,
#             ),
#         ),
#         run=RunSpec(
#             seed=0,
#             data={
#                 "test": True,
#             },
#             record_post_training_metrics=False,
#             record_times=True,
#             model_saving=ModelSavingSpec(
#                 save_initial_model=True,
#                 save_trained_model=True,
#                 save_model_epochs=[],
#                 save_epochs=[
#                     2,
#                     4,
#                 ],
#                 fixed_interval=0,
#                 fixed_threshold=0,
#                 exponential_rate=0,
#             ),
#             resume_checkpoint=TrainingExperimentCheckpoint(
#                 run_id=uuid.UUID("b0bdddc9-551e-491b-97e9-96ed6d43ceaf"),
#                 load_mask=True,
#                 load_optimizer=False,
#                 epoch=TrainingEpoch(
#                     epoch=2,
#                     model_number=0,
#                     model_epoch=2,
#                 ),
#             ),
#         ),
#     )

#     experiment_test_util.run_experiment(run)


# def test_resume():
#     job_id = uuid.UUID("b0bdddc9-551e-491b-97e9-96ed6d43ceaf")
#     resume_experiment = experiment = TrainingExperiment(
#         seed=100,
#         batch="test",
#         experiment_tags={
#             "model_family": "lenet",
#             "model_name": "lenet_relu",
#         },
#         run_tags={
#             "test": True,
#         },
#         precision="float32",
#         dataset=DatasetSpec(
#             "mnist",
#             "keras",
#             "shuffled_train_test_split",
#             0.2,
#             0.05,
#             0.0,
#         ),
#         model=LayerFactoryModel(
#             layer_factory=SequentialModel(
#                 [
#                     DenseConv.make(
#                         6,
#                         [5, 5],
#                         [1, 1],
#                         {
#                             "padding": "same",
#                             "use_bias": True,
#                             "activation": "sigmoid",
#                             "kernel_constraint": keras_kwcfg("ParameterMask"),
#                         },
#                     ),
#                     AvgPool.make([2, 2], [2, 2]),
#                     DenseConv.make(
#                         6,
#                         [5, 5],
#                         [1, 1],
#                         {
#                             "padding": "valid",
#                             "use_bias": True,
#                             "kernel_constraint": keras_kwcfg("ParameterMask"),
#                         },
#                     ),
#                     AvgPool.make([2, 2], [2, 2]),
#                     FullyConnectedNetwork(
#                         widths=[120, 84],
#                         flatten_input=True,
#                         inner=Dense.make(-1, {}),
#                     ),
#                 ]
#             )
#         ),
#         fit={
#             "batch_size": 32,
#             "epochs": 10,
#         },
#         optimizer={"class": "Adam", "learning_rate": 0.0001},
#         loss=None,
#         early_stopping=keras_kwcfg(
#             "EarlyStopping",
#             monitor="val_loss",
#             min_delta=0,
#             patience=50,
#             restore_best_weights=True,
#         ),
#         record=RunSpec(
#             post_training_metrics=True,
#             times=True,
#             model=None,
#             metrics=None,
#             model_saving=ModelSavingConfig(
#                 save_initial_model=True,
#                 save_trained_model=True,
#                 save_model_epochs=[],
#                 save_epochs=[],
#                 fixed_interval=0,
#                 fixed_threshold=0,
#                 exponential_rate=0,
#             ),
#             resume_from=TrainingExperimentCheckpoint(
#                 run_id=job_id,
#                 load_mask=True,
#                 load_optimizer=True,
#                 epoch=TrainingEpoch(
#                     epoch=5,
#                     model_number=0,
#                     model_epoch=1,
#                 ),
#             ),
#         ),
#     )
#     run_experiment(resume_experiment)


if __name__ == "__main__":
    test_mnist_lenet()
    # test_resume()
