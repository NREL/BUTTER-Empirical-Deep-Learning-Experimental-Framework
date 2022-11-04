from dmp.data.pmlb import pmlb_loader
from dmp.structure.network_module import NetworkModule
import dmp.task.aspect_test.aspect_test_utils as aspect_test_utils
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as numpy
from pmlb import fetch_data
from tensorflow import keras
import tensorflow as tf
from typing import Any, Callable, Iterable, List, Optional, Tuple
import sklearn.model_selection
import numpy.ma as ma
from functools import *
from keras import Model


def make_feedback_training_strategy():
    pass


def make_network_of_shape_factory(
    input_shape: Tuple[int, ...],
    residual_mode: Optional[str],
    input_activation: str,
    internal_activation: str,
    output_activation: str,
    make_widths: Callable[[int], List[int]],
    layer_args: dict,
):
    def network_factory(target_size):
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

        model = aspect_test_utils.make_keras_network_from_network_module(
            network)
        return delta, widths, network, model
    return network_factory


class SimplexConstraint(tf.keras.constraints.Constraint):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        w_nonneg = w * tf.cast(tf.greater_equal(w, 0.),
                               tf.keras.backend.floatx())
        return w_nonneg / (
            tf.keras.backend.epsilon() + tf.keras.backend.sqrt(
                tf.reduce_sum(
                    tf.square(w_nonneg), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'axis': self.axis}


class WeightedAdd(tf.keras.layers.Layer):
    def __init__(self, initial_weights, trainable_weights=False):
        super(WeightedAdd, self).__init__()
        initial_weights = initial_weights / numpy.sum(initial_weights)
        num_inputs = initial_weights.shape[0]
        self.input_weights = tf.Variable(
            shape=(num_inputs, 1),
            initial_value=initial_weights, trainable=trainable_weights,
            constraint=SimplexConstraint())

    def call(self, inputs):
        # normalized_weights = self.input_weights / tf.reduce_sum(self.input_weights)
        normalized_weights = self.input_weights

        acc = tf.multiply(inputs[0], normalized_weights[0])
        for i in range(1, self.input_weights.shape[0]):
            acc += tf.multiply(inputs[i], normalized_weights[i])
        return acc


Trace = dict

ActiveModels = Tuple[Model, Model, Model]

ModelFactory = Callable[[int], Tuple[int,
                                     List[int], NetworkModule, Model]]
GrowthStrategy = Callable[[ActiveModels, Trace],
                          Tuple[bool, ActiveModels]]
GrowthTrigger = Callable[[ActiveModels, Trace], bool]

CompositeModelFactory = Callable[[
    ActiveModels, Trace], ActiveModels]

# tf.keras.losses.CategoricalCrossentropy()


def direct_evaluation_composite_model_factory(
    active_models: ActiveModels,
    trace: Trace,
) -> ActiveModels:
    return active_models


def averaging_weighting_function(
    evaluation_model: Model,
    individual_models: List[Model],
    trace: Trace,
) -> numpy.ndarray:
    return numpy.ones((len(individual_models), 1), dtype=numpy.float32)


def size_weighting_function(
    evaluation_model: Model,
    individual_models: List[Model],
    trace: Trace,
) -> numpy.ndarray:
    return numpy.array([
        aspect_test_utils.count_parameters_in_keras_model(model[0])
        for model in individual_models], dtype=numpy.float32)

def make_weighted_sum_evaluation_model_factory(
    weighting_function,
    trainable_weights: bool,
    train_together: bool,
    freeze_previous_models: bool,
) -> CompositeModelFactory:
    individual_models = []

    def composite_model_factory(
        active_models: ActiveModels,
        trace: Trace,
    ) -> ActiveModels:
        nonlocal individual_models

        evaluation_model, training_model, individual_model = active_models

        if freeze_previous_models and len(individual_models) > 0:
            individual_models[-1].trainable = False

        individual_models.append(training_model)
        individual_model = training_model

        input_shape = evaluation_model.layers[0].input_shape
        # output_shape = evaluation_model.layers[-1].output_shape
        input = tf.keras.layers.Input(shape=input_shape)
        model_outputs = [model[0](input) for model in individual_models]
        initial_weights = \
            weighting_function(evaluation_model, individual_models, trace)
        initial_weights = initial_weights.reshape((len(individual_models), 1))
        output = WeightedAdd(
            initial_weights, trainable_weights=trainable_weights)(model_outputs)
        output = tf.keras.layers.Average()(model_outputs)
        evaluation_model = tf.keras.Model(input, output)

        if train_together:
            training_model = evaluation_model

        return evaluation_model, training_model, individual_model

    return composite_model_factory


def make_scheduled_growth_strategy(
    sizes: List[int],
    growth_trigger: GrowthTrigger,
    model_factory: ModelFactory,
    composite_model_factory: CompositeModelFactory,
    optimizer_config: dict,
    loss_function: Any,
) -> GrowthStrategy:
    optimizer = tf.keras.optimizers.get(optimizer_config)
    stage = 0

    def grow(
        active_models: ActiveModels,
        trace: Trace,
    ):
        nonlocal stage

        evaluation_model, training_model, individual_model = active_models
        end_training = False

        if growth_trigger(active_models, trace):
            end_training = stage >= len(sizes)
            if not end_training:
                size = sizes[stage]
                stage += 1

                remaining_parameters = size
                if evaluation_model is not None:
                    remaining_parameters -= \
                        aspect_test_utils.count_parameters_in_keras_model(
                            evaluation_model)

                delta, widths, network, training_model = model_factory(
                    remaining_parameters)

                evaluation_model, training_model, individual_model = \
                    composite_model_factory(
                        (evaluation_model,
                         training_model,
                         individual_model),
                        trace,
                    )

                training_model.compile(
                    optimizer=optimizer,
                    loss=loss_function)

                if evaluation_model is not training_model:
                    training_model.compile(
                        optimizer=optimizer,
                        loss=loss_function)

                if individual_model is not training_model:
                    individual_model.compile(
                        optimizer=optimizer,
                        loss=loss_function)

        return end_training, (evaluation_model, training_model, individual_model)
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
    training_model = None
    evaluation_model = None
    individual_model = None

    def accumulate_trace_vars(prefix, model, epoch, training_loss, test_loss, validation_loss):
        nonlocal trace
        prefix += '_'
        trace[prefix + 'parameters'].append(aspect_test_utils.count_parameters_in_keras_model(model))
        trace[prefix + 'trainable_parameters'].append(aspect_test_utils.count_trainable_parameters_in_keras_model(model))
        trace[prefix + 'non_trainable_parameters'].append(aspect_test_utils.count_non_trainable_parameters_in_keras_model(model))
        trace[prefix + 'training_loss'].append(training_loss)
        trace[prefix + 'test_loss'].append(test_loss)
        trace[prefix + 'validation_loss'].append(validation_loss)
        trace[prefix + 'epoch'].append(epoch)
        # trace[prefix + 'parameter_epochs'].append(epoch)



    while True:
        end_training, (evaluation_model, training_model, individual_model) = \
            growth_strategy((evaluation_model, training_model, individual_model), trace)
        if end_training:
            break

        accumulate_trace_vars('evaluation_model', evaluation_model, )
        accumulate_trace_vars('training_model', training_model)
        accumulate_trace_vars('individual_model', individual_model)

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
