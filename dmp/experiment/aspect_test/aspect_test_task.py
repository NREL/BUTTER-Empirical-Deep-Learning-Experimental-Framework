from abc import abstractmethod

from attr import dataclass
from task import TrainingTask

import gc
import json
import math
import os
import random
import sys
from copy import deepcopy
from functools import singledispatchmethod
from typing import Callable, Union, List, Optional

import numpy
import pandas
import tensorflow
import itertools
from keras_buoy.models import ResumableModel
from sklearn.model_selection import train_test_split
from tensorflow.keras import (
    callbacks,
    metrics,
    optimizers,
)
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras import losses, Input
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

from command_line_tools import (
    command_line_config,
    run_tools,
)
from dmp.data.logging import write_log
from dmp.data.pmlb import pmlb_loader
from dmp.data.pmlb.pmlb_loader import load_dataset
from dmp.experiment.structure.algorithm.network_json_serializer import NetworkJSONSerializer
from dmp.experiment.structure.n_add import NAdd
from dmp.experiment.structure.n_dense import NDense
from dmp.experiment.structure.n_input import NInput
from dmp.experiment.structure.network_module import NetworkModule
from dmp.jq import jq_worker

from aspect_test_utils import *

from dmp.experiment.batch.batch import CartesianBatch

@dataclass
class AspectTestTaskResult():
    pass

@dataclass
class AspectTestTaskCartesianBatch(CartesianBatch):
    """
    for task in AspectTestTaskCartesianBatch(**config)...
    """
    dataset: list[str] = ['529_pollen'],
    learning_rate: list[float] = [0.001],
    topology: list[str] = ['wide_first'],
    budget: list[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
            8388608, 16777216, 33554432],
    depth: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

    def __task_class():
        return AspectTestTask


@dataclass
class AspectTestTask(TrainingTask):
    ### Parameters
    seed: int = None
    log: str = './log'
    dataset: str = '529_pollen'
    test_split: int = 0
    activation: str = 'relu'
    optimizer: dict = {
        "class_name": "adam",
        "config": {"learning_rate": 0.001},
        }
    dataset: str = '529_pollen'
    learning_rate: float = 0.001
    topology: str = 'wide_first'
    budget: int = 32
    depth: int = 10
    epoch_scale: dict = {
        'm': 0,
        'b': numpy.log(3001),
    }
    residual_mode: str = 'none'
    rep: 0
    early_stopping: bool = False,
    run_config: dict = {
        'validation_split': .2,  # This is relative to the training set size.
        'shuffle': True,
        'epochs': 3000,
        'batch_size': 256,
        'verbose': 0
        }
    
    ### Instance Vars
    __strategy: tensorflow.distribute.Strategy

    def run(self):
        """
        Execute this task and return the result
        """

        datasets = pmlb_loader.load_dataset_index()

        if self.__strategy is None:
            self.__strategy = tensorflow.distribute.get_strategy()

        numpy.random.seed(self.seed)
        tensorflow.random.set_seed(self.seed)
        random.seed(self.seed)

        ## Load dataset
        dataset, inputs, outputs = load_dataset(datasets, self.dataset)

        num_outputs = outputs.shape[1]

        ## Network configuration
        output_activation, run_loss = compute_network_configuration(num_outputs, self.dataset)

        ## Build NetworkModule network
        delta, widths, network = find_best_layout_for_budget_and_depth(
            inputs,
            self.residual_mode,
            'relu',
            'relu',
            output_activation,
            self.budget,
            widths_factory(self.topology)(num_outputs, self.depth),
            self.depth,
            self.topology
        )

        network_structure = NetworkJSONSerializer(network)()

        print('begin reps: budget: {}, depth: {}, widths: {}, rep: {}'.format(self.budget, self.depth, self.widths, self.rep))

        ## Create Keras model from NetworkModule
        with self.__strategy.scope():
            keras_inputs, keras_output = make_model_from_network(network)

            assert len(keras_inputs) == 1, 'Wrong number of keras inputs generated'
            keras_input = keras_inputs[0]

            ## Run Keras model on dataset
            run_log = self.test_network(dataset, inputs, outputs, keras_input, keras_output, network, run_loss, widths)

        return run_log

    def test_network(self,
        dataset,
        inputs: numpy.ndarray,
        outputs: numpy.ndarray,
        keras_input,
        keras_output,
        network: NetworkModule,
        run_loss,
        widths
    ) -> dict:
        """
        test_network

        Given a fully constructed Keras network, train and test it on a given dataset using hyperparameters in config
        This function also creates log events during and after training.
        """

        # wine_quality_white__wide_first__4194304__4__16106579275625
        name = '{}__{}__{}__{}'.format(
            self.dataset,
            self.topology,
            self.budget,
            self.depth,
        )

        self.name = name

        run_name = run_tools.get_run_name(config)
        self.run_name = run_name

        depth = len(widths)
        self.depth = depth
        self.num_hidden = max(0, depth - 2)
        self.widths = widths

        run_optimizer = optimizers.get(self.optimizer)
        run_metrics = [
            # metrics.CategoricalAccuracy(),
            'accuracy',
            metrics.CosineSimilarity(),
            metrics.Hinge(),
            metrics.KLDivergence(),
            metrics.MeanAbsoluteError(),
            metrics.MeanSquaredError(),
            metrics.MeanSquaredLogarithmicError(),
            metrics.RootMeanSquaredError(),
            metrics.SquaredHinge(),
        ]

        model = Model(inputs=keras_input, outputs=keras_output)

        ## TODO: Would holding off on this step obviate the need for NetworkModule?
        model.compile(
            # loss='binary_crossentropy', # binary classification
            # loss='categorical_crossentropy', # categorical classification (one hot)
            loss=run_loss,  # regression
            optimizer=run_optimizer,
            # optimizer='rmsprop',
            # metrics=['accuracy'],
            metrics=run_metrics,
        )

        gc.collect()

        assert count_num_free_parameters(network) == count_trainable_parameters_in_keras_model(model), \
            "Wrong number of trainable parameters"

        log_data = {}
        log_data['num_weights'] = count_trainable_parameters_in_keras_model(model)
        log_data['num_inputs'] = inputs.shape[1]
        log_data['num_features'] = dataset['n_features']
        log_data['num_classes'] = dataset['n_classes']
        log_data['num_outputs'] = outputs.shape[1]
        log_data['num_observations'] = inputs.shape[0]
        log_data['task'] = dataset['Task']
        log_data['endpoint'] = dataset['Endpoint']

        run_callbacks = []
        if self.early_stopping != False:
            run_callbacks.append(callbacks.EarlyStopping(**self.early_stopping))

        run_config = self.run_config.copy()

        if self.test_split > 0:
            ## train/test/val split
            inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs,
                                                                                    test_size=self.test_split])
            run_config["validation_split"] = run_config["validation_split"] / (1 -self.test_split)

            ## Set up a custom callback to record test loss at each epoch
            ## This could potentially cause performance issues with large datasets on GPU
            class TestHistory(Callback):
                def __init__(self, x_test, y_test):
                    self.x_test = x_test
                    self.y_test = y_test

                def on_train_begin(self, logs={}):
                    self.history = {}

                def on_epoch_end(self, epoch, logs=None):
                    eval_log = self.model.evaluate(x=self.x_test, y=self.y_test, return_dict=True)
                    for k, v in eval_log.items():
                        k = "test_" + k
                        self.history.setdefault(k, []).append(v)

            test_history_callback = TestHistory(inputs_test, outputs_test)
            run_callbacks.append(test_history_callback)
        else:
            ## Just train/val split
            inputs_train, outputs_train = inputs, outputs

        if "tensorboard" in config.keys():
            run_callbacks.append(TensorBoard(
                log_dir=os.path.join(config["tensorboard"], run_name),
                # append ",self.residual_mode" to add resisual to tensorboard path
                histogram_freq=1
            ))

        if "plot_model" in config.keys():
            if not os.path.exists(config["plot_model"]):
                os.makedirs(config["plot_model"])
            tensorflow.keras.utils.plot_model(
                model,
                to_file=os.path.join(config["plot_model"], run_name + ".png"),
                show_shapes=False,
                show_dtype=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=False,
                dpi=96,
            )

        # TRAINING
        # run_config["verbose"] = 0  # This overrides verbose logging.

        ## Checkpoint Code
        if "checkpoint_epochs" in config.keys():

            assertself.test_split == 0, "Checkpointing is not compatible with test_split."

            DMP_CHECKPOINT_DIR = os.getenv("DMP_CHECKPOINT_DIR", default="checkpoints")
            if "checkpoint_dir" in config.keys():
                DMP_CHECKPOINT_DIR =self.checkpoint_dir
            if not os.path.exists(DMP_CHECKPOINT_DIR):
                os.makedirs(DMP_CHECKPOINT_DIR)

            if "jq_uuid" in config.keys():
                checkpoint_name =self.jq_uuid
            else:
                checkpoint_name = run_name

            model = ResumableModel(model,
                                save_every_epochs=config["checkpoint_epochs"],
                                to_path=os.path.join(DMP_CHECKPOINT_DIR, checkpoint_name + ".h5"))

        history = model.fit(
            x=inputs_train,
            y=outputs_train,
            callbacks=run_callbacks,
            **run_config,
        )

        if not "checkpoint_epochs" in config.keys():
            # Tensorflow models return a History object from their fit function, but ResumableModel objects returns History.history. This smooths out that incompatibility.
            history = history.history

        # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
        # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
        # model.save_weights(f"./log/weights/{run_name}.h5", save_format="h5")
        # model.save(f"./log/models/{run_name}.h5", save_format="h5")

        log_data['history'] = history

        validation_losses = numpy.array(history['val_loss'])
        best_index = numpy.argmin(validation_losses)

        log_data['iterations'] = best_index + 1
        log_data['val_loss'] = validation_losses[best_index]
        log_data['loss'] = history['loss'][best_index]

        ifself.test_split > 0:
            ## Record the history of test evals from the callback
            log_data['history'].update(test_history_callback.history)
            test_losses = numpy.array(history['test_loss'])
            log_data['test_loss'] = test_losses[best_index]

        log_data['run_name'] = run_name

        log_data['environment'] = {
            "tensorflow_version": tensorflow.__version__,
        }

        return log_data


