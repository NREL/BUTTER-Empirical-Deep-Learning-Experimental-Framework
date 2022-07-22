from dmp.data.pmlb import pmlb_loader
import dmp.task.aspect_test.aspect_test_utils as aspect_test_utils
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pmlb import fetch_data
from tensorflow import keras
import tensorflow as tf
from typing import Callable, Iterable, List, Optional, Tuple
import sklearn.model_selection
import numpy.ma as ma
from functools import *

   

def make_feedback_training_strategy():
    pass

def make_network_of_shape_generator(
    input_shape: Tuple[int,...],
    residual_mode: Optional[str],
    input_activation : str,
    internal_activation : str,
    output_activation : str,
    make_widths: Callable[[int], List[int]],
    layer_args: dict,
):
    def make_network(target_size):
        delta, widths, network = \
            aspect_test_utils.find_best_layout_for_budget_and_depth(
                input_shape,
                residual_mode,
                input_activation,
                internal_activation,
                output_activation,
                target_size,
                make_widths,
                layer_args,
            )

        model = aspect_test_utils.make_keras_network_from_network_module(network)
        return delta, widths, network, model
    return make_network

def make_scheduled_growth_strategy(sizes, growth_trigger, network_factory, optimizer_config):
    models = []
    def grow(evaluation_model, training_model, ):
        nonlocal models
        stage = len(models)
        size = sizes[stage]

        remaining_parameters = size
        if evaluation_model is not None:
            remaining_parameters -= \
                aspect_test_utils.count_parameters_in_keras_model(evaluation_model)

        network_factory()

        new_model = make_model_of_size(
            inputs,
            remaining_parameters,
            make_widths,
            output_activation,
            layer_args,
        )
        models.append((new_model, None))

        optimizer = tf.keras.optimizers.get(optimizer_config)
        new_model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.CategoricalCrossentropy())

        return evaluation_model, training_model
    return grow


def train_grow(dataset, growth_strategy, make_widths, optimizer_config, freeze_old=True, val_split=None):
    dataset_series, inputs, outputs, train_inputs, train_outputs, val_inputs, val_outputs, test_inputs, test_outputs, output_activation, run_loss = dataset

    layer_args = {'kernel_initializer': 'he_normal'}

    trace = {m: [] for m in (
        'loss',
        'val_loss',
        'test_loss',
        'size',
        'effort',
        'total_effort',
        'epoch',
        'i_loss',
        'i_val_loss',
        'i_test_loss',
    )}

    epoch = 0
    effort = 0
    free_effort = 0
    models = []
    model = None
    while True:
        model, training_model = growth_strategy(model, training_model, trace)
        optimizer = tf.keras.optimizers.get(optimizer_config)

        new_model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.CategoricalCrossentropy())
        if new_model != model:
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.CategoricalCrossentropy())

        # print(model.summary())

        trace['size'].append(total_model_size)

        # print('aggregate assesment...')
        trace['loss'].append(evaluate_model(model, train_inputs,
                                            train_outputs))
        trace['val_loss'].append(
            evaluate_model(model, val_inputs, val_outputs))
        trace['test_loss'].append(
            trace['val_loss'][-1] if val_inputs is test_inputs else evaluate_model(model, test_inputs, test_outputs))

        # print('individual assesment...')
        # trace['i_loss'].append(
        #     evaluate_model(new_model, train_inputs, train_outputs))
        # trace['i_val_loss'].append(
        #     evaluate_model(new_model, val_inputs, val_outputs))
        # trace['i_test_loss'].append(
        #     trace['i_val_loss'][-1] if val_inputs is test_inputs else evaluate_model(new_model, test_inputs, test_outputs))

        trace['effort'].append(free_effort)
        trace['total_effort'].append(effort)
        trace['epoch'].append(epoch)

        epochs_at_stage = 0
        while True:
            end_epoch = epochs_at_stage + 1
            epoch += 1
            effort += total_model_size
            free_effort += new_model_size

            fit_result = model.fit(train_inputs, train_outputs,
                                   validation_data=(val_inputs, val_outputs),
                                   initial_epoch=epochs_at_stage,
                                   epochs=end_epoch,
                                   verbose=0,
                                   )
            epochs_at_stage = end_epoch
            # print(fit_result.history)

            trace['loss'].append(fit_result.history['loss'][0])
            trace['val_loss'].append(fit_result.history['val_loss'][0])
            trace['test_loss'].append(
                trace['val_loss'][-1] if val_inputs is test_inputs else evaluate_model(model, test_inputs, test_outputs))

            # print('individual assesment...')
            # trace['i_loss'].append(
            #     evaluate_model(new_model, train_inputs, train_outputs))
            # trace['i_val_loss'].append(
            #     evaluate_model(new_model, val_inputs, val_outputs))
            # trace['i_test_loss'].append(
            #     trace['i_val_loss'][-1] if val_inputs is test_inputs else evaluate_model(new_model, test_inputs, test_outputs))

            trace['size'].append(size)
            trace['epoch'].append(epoch)
            trace['effort'].append(free_effort)
            trace['total_effort'].append(effort)

            if growth_trigger(model, epochs_at_stage, trace):
                break

        if freeze_old:
            new_model.trainable = False  # freeze old models...

    return model, trace