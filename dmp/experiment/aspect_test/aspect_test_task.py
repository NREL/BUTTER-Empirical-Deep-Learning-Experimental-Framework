import dataclasses
from dataclasses import field
from os import environ
import time
from typing import Type

from attr import dataclass

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


@dataclass
class AspectTestTaskRunVariables():
    """
    Namespace which is created when a task is run and contains state variables needed to run the task.
    This class is not, in general, logged to the database.
    """
    tf_strategy: tensorflow.distribute.Strategy = None
    keras_model: tensorflow.keras.Model
    keras_loss: tensorflow.keras.losses
    result: AspectTestTaskResult = AspectTestTaskResult()
    network_module: NetworkModule
    dataset: pandas.Series
    dataset_inputs: numpy.ndarray
    dataset_outputs: numpy.ndarray




@dataclass
class AspectTestTask(Task):
    # Parameters
    seed: int
    # log: str = './log'
    dataset: str
    # test_split: int = 0
    input_activation: str
    internal_activation: str
    optimizer: dict
    # learning_rate: float = None
    topology: str
    residual_mode: str
    budget: int
    depth: int
    # epoch_scale: dict
    # rep: int
    early_stopping: Optional[dict]
    validation_split: float
    run_config: dict
    checkpoint_epochs: Optional[int]

    id: Optional[uuid.UUID] = None

    # Instance Vars (initialized on execution)
    tf_strategy: Optional[tensorflow.distribute.Strategy] = None
    keras_model: Optional[tensorflow.keras.Model] = None
    keras_loss: Optional[tensorflow.keras.losses] = None
    result: Optional[AspectTestTaskResult] = AspectTestTaskResult()
    network_module: Optional[NetworkModule] = None
    dataset: Optional[pandas.Series] = None
    dataset_inputs: Optional[numpy.ndarray] = None
    dataset_outputs: Optional[numpy.ndarray] = None

    def __call__(self):
        # Step 1. Load dataset

        datasets = pmlb_loader.load_dataset_index()
        self._run.dataset, self._run.dataset_inputs, self._run.dataset_outputs = load_dataset(
            datasets, self.dataset)

        # Step 2. Configure hardware and set random seed

        if self._run.tf_strategy is None:
            self._run.tf_strategy = tensorflow.distribute.get_strategy()

        if self.seed is None:
            self.seed = time.time_ns()

        numpy.random.seed(self.seed)
        tensorflow.random.set_seed(self.seed)
        random.seed(self.seed)

        # Step 3. Generate neural network architecture

        num_outputs = self._run.dataset_outputs.shape[1]
        output_activation, self._run.keras_loss = compute_network_configuration(
            num_outputs, self._run.dataset)

        delta, widths, self._run.network_module = find_best_layout_for_budget_and_depth(
            self._run.dataset_inputs,
            self.residual_mode,
            self.input_activation,
            self.internal_activation,
            output_activation,
            self.budget,
            widths_factory(self.topology)(num_outputs, self.depth),
            self.depth,
            self.topology
        )

        # network_structure = NetworkJSONSerializer(network)()

        print('begin reps: budget: {}, depth: {}, widths: {}, rep: {}'.format(self.budget, self.depth, self.widths,
                                                                              self.rep))

        # Step 4: Create and execute network using Keras

        with self._run.tf_strategy.scope():

            # Build Keras model
            self._run.keras_model = build_keras_network(
                self._run.network_module)
            assert len(
                self._run.keras_model.inputs) == 1, 'Wrong number of keras inputs generated'

            # Execute Keras model
            self._run.result = self.test_keras_network()

    def log_result(self) -> None:
        pass
        engine, session = _connect()
        for record_class in [BaseRecord, ValLossRecord, HistoryRecord]:
            record_element = self.make_dataclass_from_dict(
                self._run.result, record_class)
            session.add(record_element)
        session.commit()
        # TODO: finish this guy

    def make_dataclass_from_dict(self, source: {}, cls: Type) -> any:
        keys = {f.name for f in dataclasses.fields(
            BaseRecord) if not f.name.startswith('__')}
        return cls(**{k: v for k, v in source if k in keys})

    def test_keras_network(self, tensorboard=False, plot_model=False) -> None:
        """
        test_keras_network

        Given a fully constructed Keras network, train and test it on a given dataset using hyperparameters in config
        This function also creates log events during and after training.
        """

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

        self._run.keras_model.compile(
            # loss='binary_crossentropy', # binary classification
            # loss='categorical_crossentropy', # categorical classification (one hot)
            loss=self._run.keras_loss,  # regression
            optimizer=optimizers.get(self.optimizer),
            # optimizer='rmsprop',
            # metrics=['accuracy'],
            metrics=run_metrics,
        )

        gc.collect()

        assert count_num_free_parameters(self._run.network_module) == count_trainable_parameters_in_keras_model(self._run.keras_model), \
            "Wrong number of trainable parameters"

        # Configure Keras Callbacks

        run_callbacks = []

        if self.early_stopping is not None:
            run_callbacks.append(
                callbacks.EarlyStopping(**self.early_stopping))

        if tensorboard:
            run_callbacks.append(TensorBoard(
                log_dir=os.path.join(tensorboard, self.uuid),
                # append ",self.residual_mode" to add resisual to tensorboard path
                histogram_freq=1
            ))

        if plot_model:
            if not os.path.exists(plot_model):
                os.makedirs(plot_model)
            tensorflow.keras.utils.plot_model(
                self._run.keras_model,
                to_file=os.path.join(plot_model, self.uuid + ".png"),
                show_shapes=False,
                show_dtype=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=False,
                dpi=96,
            )

        # Checkpoint Code
        if self.checkpoint_epochs is not None and self.checkpoint_epochs > 0:

            # assert self.test_split == 0, "Checkpointing is not compatible with test_split."

            DMP_CHECKPOINT_DIR = os.getenv(
                "DMP_CHECKPOINT_DIR", default="checkpoints")
            if not os.path.exists(DMP_CHECKPOINT_DIR):
                os.makedirs(DMP_CHECKPOINT_DIR)

            save_path = os.path.join(DMP_CHECKPOINT_DIR,
                                     self.uuid + ".h5")

            self._run.keras_model = ResumableModel(self._run.keras_model,
                                                   save_every_epochs=self.checkpoint_epochs,
                                                   to_path=save_path)

        # Split data into train and validation sets manually. Keras does not shuffle the validation set by default
        x_train, x_val, y_train, y_val = train_test_split(self._run.dataset_inputs,
                                                          self._run.dataset_outputs,
                                                          shuffle=True,
                                                          test_size=self.validation_split)
        # Enter Keras training loop
        history = self._run.keras_model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            callbacks=run_callbacks,
            **self.run_config,
        )

        # Tensorflow models return a History object from their fit function, but ResumableModel objects returns History.history. This smooths out that incompatibility.
        if self.checkpoint_epochs is None or self.checkpoint_epochs == 0:
            history = history.history

        # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
        # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
        # model.save_weights(f"./log/weights/{run_name}.h5", save_format="h5")
        # model.save(f"./log/models/{run_name}.h5", save_format="h5")

        result = self._run.result

        result.num_weights = count_trainable_parameters_in_keras_model(
            self._run.keras_model)
        result.num_inputs = self._run.dataset_inputs.shape[1]
        result.num_outputs = self._run.dataset_outputs.shape[1]
        result.num_features = self._run.dataset['n_features']
        result.num_classes = self._run.dataset['n_classes']
        result.num_observations = self._run.dataset_inputs.shape[0]
        result.task = self._run.dataset['Task']
        result.endpoint = self._run.dataset['Endpoint']
        result.history = history

        validation_losses = numpy.array(history['val_loss'])
        best_index = numpy.argmin(validation_losses)

        result.iterations = best_index + 1
        result.val_loss = validation_losses[best_index]
        result.loss = history['loss'][best_index]

        result.environment = {
            "tensorflow_version": tensorflow.__version__,
        }

        # result.run_name = run_name # I think this is deprecated

        return
