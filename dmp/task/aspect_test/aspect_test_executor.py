import dataclasses
from dataclasses import dataclass, field
from os import environ
import time
from typing import Type


from .aspect_test_task import AspectTestTask
from .aspect_test_result import AspectTestResult
from sklearn.model_selection import train_test_split

from .aspect_test_utils import *
from dmp.batch.batch import CartesianBatch
from dmp.task.task import Task
from dmp.record.base_record import BaseRecord
from dmp.record.history_record import HistoryRecord
from dmp.record.val_loss_record import ValLossRecord

import pandas
import numpy
import uuid

from dmp.logging.result_logger import ResultLogger


_datasets = pmlb_loader.load_dataset_index()


@dataclass
class AspectTestExecutor(AspectTestTask):
    '''
    '''

    parent: AspectTestTask = None
    output_activation: Optional[str] = None
    job_id: Optional[uuid.UUID] = None
    tensorflow_strategy: Optional[tensorflow.distribute.Strategy] = None
    keras_model: Optional[tensorflow.keras.Model] = None
    run_loss: Optional[tensorflow.keras.losses] = None
    network_module: Optional[NetworkModule] = None
    dataset: Optional[pandas.Series] = None
    inputs: Optional[numpy.ndarray] = None
    outputs: Optional[numpy.ndarray] = None

    def __call__(self) -> None:
        # Configure hardware
        if self.tensorflow_strategy is None:
            self.tensorflow_strategy = tensorflow.distribute.get_strategy()

        # Set random seeds
        self.seed = set_random_seeds(self.seed)

        # Load dataset
        self.dataset, self.inputs, self.outputs =  \
            load_dataset(_datasets, self.dataset)

        # prepare dataset shuffle, split, and label noise:
        prepared_config = prepare_dataset(
            self.validation_split_method,
            self.label_noise,
            self.run_config,
            self.dataset['Task'],
            self.inputs,
            self.outputs,
        )

        # Generate neural network architecture
        num_outputs = self.outputs.shape[1]
        self.output_activation, self.run_loss = \
            compute_network_configuration(num_outputs, self.dataset)

        # Build NetworkModule network
        delta, widths, self.network_module = find_best_layout_for_budget_and_depth(
            self.inputs,
            self.residual_mode,
            self.input_activation,
            self.activation,
            self.output_activation,
            self.budget,
            widths_factory(self.topology)(num_outputs, self.depth),
            self.depth,
            self.topology
        )

        print('begin reps: budget: {}, depth: {}, widths: {}, rep: {}'.format(self.budget, self.depth, self.widths,
                                                                              self.rep))

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

            run_optimizer = tensorflow.keras.optimizers.get(self.optimizer)

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
            if self.early_stopping is not None:
                run_callbacks.append(
                    callbacks.EarlyStopping(**self.early_stopping))

            # optionally enable checkpoints
            if self.save_every_epochs is not None and self.save_every_epochs > 0:
                DMP_CHECKPOINT_DIR = os.getenv(
                    'DMP_CHECKPOINT_DIR', default='checkpoints')
                if not os.path.exists(DMP_CHECKPOINT_DIR):
                    os.makedirs(DMP_CHECKPOINT_DIR)

                save_path = os.path.join(
                    DMP_CHECKPOINT_DIR, self.job_id + '.h5')

                self.keras_model = ResumableModel(
                    self.keras_model,
                    save_every_epochs=self.save_every_epochs,
                    to_path=save_path)

            # fit / train model
            history = self.keras_model.fit(
                callbacks=run_callbacks,
                **prepared_config,
            )

            # Tensorflow models return a History object from their fit function,
            # but ResumableModel objects returns History.history. This smooths
            # out that incompatibility.
            if self.save_every_epochs is None or self.save_every_epochs == 0:
                history = history.history

            # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
            # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
            # model.save_weights(f'./log/weights/{run_name}.h5', save_format='h5')
            # model.save(f'./log/models/{run_name}.h5', save_format='h5')

            runtime_parameters = {
                'tensorflow_version': tensorflow.__version__,
            }

            # return the result record
            return runtime_parameters, history
