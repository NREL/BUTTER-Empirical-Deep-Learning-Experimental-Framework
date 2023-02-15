import sys

from jobqueue.job import Job
import numpy
import pandas
from tensorflow.python.framework.ops import re

from dmp import jobqueue_interface
from dmp.layer.flatten import Flatten
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import WidthScaler
from dmp.task.experiment.training_experiment.experiment_record_settings import ExperimentRecordSettings
from dmp.worker import Worker
from dmp.keras_interface.keras_utils import make_keras_kwcfg
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import OverlayTransfer

sys.path.insert(0, './')

import tensorflow
import dmp.jobqueue_interface.worker
import pytest

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.layer.dense import Dense
from dmp.model.dense_by_size import DenseBySize

from dmp.task.experiment.training_experiment.training_experiment import TrainingExperiment
from pprint import pprint

from dmp.marshaling import marshal

strategy = dmp.jobqueue_interface.worker.make_strategy(1, 1, 0, 8192)
worker = Worker(
    None,
    None,
    None,
    strategy,
    {},
)  # type: ignore


def run_experiment(experiment):
    results = experiment(worker, Job())
    print('experiment_attrs\n')
    pprint(results.experiment_attrs)
    print('experiment_properties\n')
    pprint(results.experiment_properties)
    print('run_data\n', results.run_data)
    print('run_history\n', results.run_history)
    print('run_extended_history\n', results.run_extended_history)
    return results


def test_simple():
    experiment = TrainingExperiment(
        seed=0,
        batch='test',
        precision='float32',
        dataset=DatasetSpec(
            # 'titanic',
            # 'pmlb',
            'GaussianClassificationDataset_2_10_100',
            # # 'GaussianRegressionDataset_20_100',
            'synthetic',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            input=None,
            output=None,
            shape='rectangle',
            size=16384,
            depth=4,
            search_method='integer',
            inner=Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit={
            'batch_size': 256,
            'epochs': 5,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.001
        },
        loss=None,
        early_stopping=None,
        record=ExperimentRecordSettings(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
        ),
    )

    run_experiment(experiment)


def test_mnist():
    width = int(2**4)

    experiment = TrainingExperiment(
        seed=0,
        batch='test',
        precision='float32',
        dataset=DatasetSpec(
            'mnist',
            'keras',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        # model=DenseBySize(
        #     input=None,
        #     output=None,
        #     shape='rectangle',
        #     size=2**20,
        #     depth=3,
        #     search_method='integer',
        #     inner=Dense.make(-1, {
        #         'activation': 'relu',
        #         'kernel_initializer': 'GlorotUniform',
        #     }),
        # ),
        model=CNNStack(
            input=None,
            output=None,
            num_stacks=2,
            cells_per_stack=1,
            stem='conv_5x5_1x1_same',
            downsample='max_pool_2x2_2x2_same',
            cell='conv_5x5_1x1_same',
            final=FullyConnectedNetwork(
                input=None,
                output=None,
                widths=[width * 2, width * 2],
                residual_mode='none',
                flatten_input=True,
                inner=Dense.make(-1, {}),
            ),
            stem_width=width,
            stack_width_scale_factor=1.0,
            downsample_width_scale_factor=1.0,
            cell_width_scale_factor=1.0,
        ),
        # model=CNNStack(
        #     input=None,
        #     output=None,
        #     num_stacks=0,
        #     cells_per_stack=0,
        #     stem='conv_5x5_1x1_same',
        #     downsample='max_pool_2x2_2x2_valid',
        #     cell='conv_5x5_1x1_same',
        #     final=Dense.make(
        #         width * 2,
        #         {},
        #         [
        #             Dense.make(
        #                 width * 2,
        #                 {},
        #                 [Flatten()],
        #             )
        #         ],
        #     ),
        #     stem_width=width,
        #     stack_width_scale_factor=1.0,
        #     downsample_width_scale_factor=1.0,
        #     cell_width_scale_factor=1.0,
        # ),
        # model=CNNStack(
        #     input=None,
        #     output=None,
        #     num_stacks=0,
        #     cells_per_stack=1,
        #     stem='conv_5x5_1x1_same',
        #     downsample='max_pool_2x2_2x2_same',
        #     cell='conv_5x5_1x1_same',
        #     final=Flatten(
        #         {},
        #         [MaxPool.make((2, 2), (2, 2))],
        #     ),
        #     stem_width=width,
        #     stack_width_scale_factor=1.0,
        #     downsample_width_scale_factor=1.0,
        #     cell_width_scale_factor=1.0,
        # ),
        fit={
            'batch_size': 256,
            'epochs': 1,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.0001
        },
        loss=None,
        early_stopping=make_keras_kwcfg(
            'EarlyStopping',
            monitor='val_loss',
            min_delta=0,
            patience=50,
            restore_best_weights=True,
        ),
        record=ExperimentRecordSettings(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
        ),
    )

    run_experiment(experiment)


def test_growth_experiment():
    experiment = GrowthExperiment(
        seed=0,
        batch='test',
        precision='float32',
        dataset=DatasetSpec(
            'titanic',
            'pmlb',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            input=None,
            output=None,
            shape='rectangle',
            size=4096,
            depth=3,
            search_method='integer',
            inner=Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit={
            'batch_size': 32,
            'epochs': 100,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.001
        },
        loss=None,
        early_stopping=None,
        record=ExperimentRecordSettings(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
        ),
        growth_trigger=make_keras_kwcfg(
            'ProportionalStopping',
            restore_best_weights=True,
            monitor='val_loss',
            min_delta=0.005,
            patience=3,
            verbose=1,
            mode='min',
            baseline=None,
            # start_from_epoch=0,
        ),
        # growth_trigger=None,
        scaling_method=WidthScaler(),
        transfer_method=OverlayTransfer(),
        growth_scale=2.0,
        initial_size=4,
        max_epochs_per_stage=300,
        max_equivalent_epoch_budget=1000,
    )

    run_experiment(experiment)


def test_from_optimizer():
    '''

    

    

"activation","relu"
"activity_regularizer",
"batch","optimizer_energy_1_cpu"
"batch_size",128
"bias_regularizer",
"dataset","mnist"
"depth",2
"early_stopping",
"epochs",3000
"input_activation","relu"
"kernel_regularizer",
"label_noise",0
"learning_rate",0.00001
"optimizer","SGD"
"optimizer.config.momentum",0.9
"optimizer.config.nesterov",false
"output_activation","softmax"
"python_version","3.9.10"
"run_config.shuffle",true
"shape","rectangle"
"size",131072
"task","AspectTestTask"
"task_version",3
"tensorflow_version","2.8.0"
"test_split",0.2
"test_split_method","shuffled_train_test_split"

{
  "": "AspectTestTask",
  "seed": 1666301679,
  "run_config": {
    "verbose": 0,
  },
  "save_every_epochs": null,
}
    '''
    experiment = TrainingExperiment(
        seed=0,
        batch='test',
        precision='float32',
        dataset=DatasetSpec(
            'banana',
            'pmlb',
            'shuffled_train_test_split',
            0.2,
            0.05,
            0.0,
        ),
        model=DenseBySize(
            None,
            None,
            'exponential',
            16384,
            3,
            'integer',
            Dense.make(-1, {
                'activation': 'relu',
                'kernel_initializer': 'GlorotUniform',
            }),
        ),
        fit={
            'batch_size': 16,
            'epochs': 3,
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.0001
        },
        loss=None,
        early_stopping=None,
        record=ExperimentRecordSettings(
            post_training_metrics=True,
            times=True,
            model=None,
            metrics=None,
        ),
    )

    run_experiment(experiment)


asdf = {
    '201_pol': (26, 11),
    '294_satellite_image': (36, 6),
    '505_tecator': (124, 1),
    '529_pollen': (4, 1),
    '537_houses': (8, 1),
    '537_houses': (784, 1),
    'adult': (8, 1),
    'adult': (81, 1),
    'banana': (2, 1),
    'connect_4': (126, 3),
    'mnist': (784, 10),
    'nursery': (26, 4),
    'nursery': (784, 4),
    'poker': (14, 10),
    'poker': (19, 10),
    'poker': (85, 10),
    'poker': (86, 10),
    'poker': (87, 10),
    'sleep': (141, 5),
    'splice': (287, 3),
    'wine_quality_white': (11, 7),
}
x = {
    '201_pol': ([26], (11, )),
    '294_satellite_image': ([36], (6, )),
    '505_tecator': ([124], (1, )),
    '529_pollen': ([4], (1, )),
    '537_houses': ([8], (1, )),
    'adult': ([81], (1, )),
    'banana': ([2], (1, )),
    'connect_4': ([126], (3, )),
    'mnist': ([784], (10, )),
    'nursery': ([26], (4, )),
    'sleep': ([141], (5, )),
    'splice': ([287], (3, )),
    'wine_quality_white': ([11], (7, ))
}

# def test_get_sizes():
#     mapping = {}
#     for dataset_name in [
#             '201_pol',
#             '294_satellite_image',
#             '505_tecator',
#             '529_pollen',
#             '537_houses',
#             'adult',
#             'banana',
#             'connect_4',
#             'mnist',
#             'nursery',
#             # 'poker',
#             'sleep',
#             'splice',
#             'wine_quality_white',
#     ]:
#         experiment = TrainingExperiment(
#             seed=0,
#             batch='test',
#             precision='float32',
#             dataset=DatasetSpec(
#                 dataset_name,
#                 'pmlb',
#                 'shuffled_train_test_split',
#                 0.2,
#                 0.05,
#                 0.0,
#             ),
#             model=DenseBySize(
#                 input=None,
#                 output=None,
#                 shape='rectangle',
#                 size=16384,
#                 depth=3,
#                 search_method='integer',
#                 inner=Dense.make(-1, {
#                     'activation': 'relu',
#                     'kernel_initializer': 'GlorotUniform',
#                 }),
#             ),
#             fit={
#                 'batch_size': 16,
#                 'epochs': 1,
#             },
#             optimizer={
#                 'class': 'Adam',
#                 'learning_rate': 0.0001
#             },
#             loss=None,
#             early_stopping=None,
#             record=ExperimentRecordSettings(
#                 post_training_metrics=True,
#                 times=True,
#                 model=None,
#                 metrics=None,
#             ),
#         )

#         worker = Worker(
#             None,
#             None,
#             strategy,
#             {},
#         )  # type: ignore

#         results = experiment(worker, Job())
#         print(f"dataset {dataset_name}")
#         print(
#             f"experiment.model.input.shape {experiment.model.input['shape']}")
#         print(
#             f"experiment.model.input.computed_shape {experiment.model.input.computed_shape}"
#         )
#         print(
#             f"experiment.model.output.units {experiment.model.output['units']}"
#         )
#         print(
#             f"experiment.model.output.computed_shape {experiment.model.output.computed_shape}"
#         )
#         mapping[dataset_name] = (experiment.model.input['shape'],
#                                  (experiment.model.output['units'], ))
#         # pprint(marshal.marshal(results), indent=1)

#     pprint(mapping)

# test_growth_experiment()
# test_simple()
test_mnist()
# test_from_optimizer()
# test_get_sizes()
