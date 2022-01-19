import dataclasses
from dataclasses import field
from os import environ
import time
from typing import Type

from attr import dataclass

from dmp.experiment.task.aspect_test_task import AspectTestTask
from sklearn.model_selection import train_test_split

from .aspect_test_utils import *
from dmp.experiment.batch.batch import CartesianBatch
from dmp.experiment.task.task import Task
from dmp.record.base_record import BaseRecord
from dmp.record.history_record import HistoryRecord
from dmp.record.val_loss_record import ValLossRecord

import pandas
import numpy
import uuid


@dataclass
class AspectTestTaskResult():
    num_weights: int
    num_inputs: int
    num_features: int
    num_classes: int
    num_outputs: int
    num_observations: int
    task: str
    endpoint: str
    history: list
    iterations: int
    val_loss: float
    loss: float
    test_loss: float
    environment: dict
    run_name: str


_datasets = pmlb_loader.load_dataset_index()


class AspectTest():
    '''
    Namespace which is created when a task is run and contains state variables needed to run the task.
    This class is not, in general, logged to the database.
    '''
    task: AspectTestTask
    job_id: uuid.UUID
    tensorflow_strategy: Optional[tensorflow.distribute.Strategy] = None
    keras_model: Optional[tensorflow.keras.Model] = None
    run_loss: Optional[tensorflow.keras.losses] = None
    result: AspectTestTaskResult = AspectTestTaskResult()
    network_module: Optional[NetworkModule] = None
    dataset: Optional[pandas.Series] = None
    inputs: Optional[numpy.ndarray] = None
    outputs: Optional[numpy.ndarray] = None

    def __init__(self, task: AspectTestTask, job_id: uuid.UUID) -> None:
        self.task.task = task
        self.task.job_id = job_id

        # Configure hardware
        if self.tensorflow_strategy is None:
            self.tensorflow_strategy = tensorflow.distribute.get_strategy()

        # Set random seeds
        self.task.seed = set_random_seeds(self.task.seed)

        # Load dataset
        self.dataset, self.inputs, self.outputs =  \
            load_dataset(_datasets, self.task.dataset)

        # prepare dataset shuffle, split, and label noise:
        prepare_dataset(
            self.task.validation_split_method,
            self.task.label_noise,
            self.task.run_config,
            self.dataset['Task'],
            self.inputs,
            self.outputs,
        )

        # Generate neural network architecture
        num_outputs = self.outputs.shape[1]
        output_activation, self.run_loss = \
            compute_network_configuration(num_outputs, self.dataset)

        # Build NetworkModule network
        delta, widths, self.network_module = find_best_layout_for_budget_and_depth(
            self.inputs,
            self.task.residual_mode,
            self.task.input_activation,
            self.task.internal_activation,
            output_activation,
            self.task.budget,
            widths_factory(self.task.topology)(num_outputs, self.task.depth),
            self.task.depth,
            self.task.topology
        )

        print('begin reps: budget: {}, depth: {}, widths: {}, rep: {}'.format(self.task.budget, self.task.depth, self.task.widths,
                                                                              self.task.rep))

        # Create and execute network using Keras
        with self.tensorflow_strategy.scope():
            # Build Keras model
            self.keras_model = make_keras_network_from_network_module(
                self.network_module)
            assert len(
                self.keras_model.inputs) == 1, 'Wrong number of keras inputs generated'

            # Compile Keras Model
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

            run_optimizer = tensorflow.keras.optimizers.get(
                self.task.optimizer)
            self.keras_model.compile(
                # loss='binary_crossentropy', # binary classification
                # loss='categorical_crossentropy', # categorical classification (one hot)
                loss=self.run_loss,  # regression
                optimizer=run_optimizer,
                # optimizer='rmsprop',
                # metrics=['accuracy'],
                metrics=run_metrics,
            )

            assert count_num_free_parameters(self.network_module) == count_trainable_parameters_in_keras_model(self.keras_model), \
                'Wrong number of trainable parameters'

            # Configure Keras Callbacks
            run_callbacks = []
            if self.task.early_stopping is not None:
                run_callbacks.append(
                    callbacks.EarlyStopping(**self.task.early_stopping))

            # optionally enable checkpoints
            if self.task.checkpoint_epochs is not None and self.task.checkpoint_epochs > 0:
                DMP_CHECKPOINT_DIR = os.getenv(
                    'DMP_CHECKPOINT_DIR', default='checkpoints')
                if not os.path.exists(DMP_CHECKPOINT_DIR):
                    os.makedirs(DMP_CHECKPOINT_DIR)

                save_path = os.path.join(
                    DMP_CHECKPOINT_DIR, self.job_id + '.h5')

                self.keras_model = ResumableModel(
                    self.keras_model,
                    save_every_epochs=self.task.checkpoint_epochs,
                    to_path=save_path)

            # fit / train model
            history = self.keras_model.fit(
                callbacks=run_callbacks,
                **self.task.run_config,
            )

            # Tensorflow models return a History object from their fit function, but ResumableModel objects returns History.history. This smooths out that incompatibility.
            if self.task.checkpoint_epochs is None or self.task.checkpoint_epochs == 0:
                history = history.history

            # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
            # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
            # model.save_weights(f'./log/weights/{run_name}.h5', save_format='h5')
            # model.save(f'./log/models/{run_name}.h5', save_format='h5')

            result = self.result

            result.num_weights = count_trainable_parameters_in_keras_model(
                self.keras_model)
            result.num_inputs = self.inputs.shape[1]
            result.num_outputs = self.outputs.shape[1]
            result.num_features = self.dataset['n_features']
            result.num_classes = self.dataset['n_classes']
            result.num_observations = self.inputs.shape[0]
            result.task = self.dataset['Task']
            result.endpoint = self.dataset['Endpoint']
            result.history = history

            validation_losses = numpy.array(history['val_loss'])
            best_index = numpy.argmin(validation_losses)

            result.iterations = best_index + 1
            result.val_loss = validation_losses[best_index]
            result.loss = history['loss'][best_index]

            result.environment = {
                'tensorflow_version': tensorflow.__version__,
            }
            

    # def log_result(self) -> None:
    #     '''

    #     config['name'] = name

    #     run_name = run_tools.get_run_name(config)
    #     config['run_name'] = run_name

    #     depth = len(widths)
    #     config['depth'] = depth
    #     config['num_hidden'] = max(0, depth - 2)
    #     config['widths'] = widths


    #     # config['widths'] = widths
    #     # config['network_structure'] = NetworkJSONSerializer(network)()
    #     # network_structure = NetworkJSONSerializer(network)()

    #     log_data = {'config': config}

    #     log_data['num_weights'] = count_trainable_parameters_in_keras_model(model)
    #     log_data['num_inputs'] = inputs.shape[1]
    #     log_data['num_features'] = dataset['n_features']
    #     log_data['num_classes'] = dataset['n_classes']
    #     log_data['num_outputs'] = outputs.shape[1]
    #     log_data['num_observations'] = inputs.shape[0]
    #     log_data['task'] = dataset['Task']
    #     log_data['endpoint'] = dataset['Endpoint']
    #     '''
    #     pass
    #     engine, session = _connect()
    #     for record_class in [BaseRecord, ValLossRecord, HistoryRecord]:
    #         record_element = self.task.make_dataclass_from_dict(
    #             self.result, record_class)
    #         session.add(record_element)
    #     session.commit()
    #     # TODO: finish this guy

    # def make_dataclass_from_dict(self, source: {}, cls: Type) -> any:
    #     keys = {f.name for f in dataclasses.fields(
    #         BaseRecord) if not f.name.startswith('__')}
    #     return cls(**{k: v for k, v in source if k in keys})
