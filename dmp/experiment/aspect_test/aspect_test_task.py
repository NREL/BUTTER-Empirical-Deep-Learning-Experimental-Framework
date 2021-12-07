import dataclasses
from os import environ
import time
from typing import Type

from attr import dataclass

from aspect_test_utils import *
from dmp.experiment.batch.batch import CartesianBatch
from dmp.experiment.task.task import Task
from dmp.record.base_record import BaseRecord
from dmp.record.history_record import HistoryRecord
from dmp.record.val_loss_record import ValLossRecord

import uuid


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
    result: AspectTestTaskResult = AspectTestTaskResult()


@dataclass
class AspectTestTask(Task):
    ### Parameters
    uuid: str = None
    seed: int = None
    log: str = './log'
    dataset: str = '529_pollen'
    test_split: int = 0
    input_activation: str = 'relu'
    internal_activation: str = 'relu'
    optimizer: dict = {
        "class_name": "adam",
        "config": {"learning_rate": 0.001},
    }
    learning_rate: float = 0.001
    topology: str = 'wide_first'
    budget: int = 32
    depth: int = 10
    epoch_scale: dict = {
        'm': 0,
        'b': numpy.log(3001),
    }
    residual_mode: str = 'none'
    rep: int = 0
    early_stopping: bool = False,
    run_config: dict = {
        'validation_split': .2,  # This is relative to the training set size.
        'shuffle': True,
        'epochs': 3000,
        'batch_size': 256,
        'verbose': 0
    }
    checkpoint_epochs: int = 0

    ### Instance Vars
    _run: AspectTestTaskRunVariables = AspectTestTaskRunVariables()

    def __call__(self):
        
        if self.uuid is None:
            # wine_quality_white__wide_first__4194304__4__16106579275625
            self.uuid = '{}__{}__{}__{}__{}'.format(
                self.dataset,
                self.topology,
                self.budget,
                self.depth,
                uuid.uuid4()
            )

        self.run()
        self.log_result()

    def run(self):
        """
        Execute this task and return the result
        """

        ## Step 1. Load dataset

        datasets = pmlb_loader.load_dataset_index()
        dataset, inputs, outputs = load_dataset(datasets, self.dataset)

        ## Step 2. Configure hardware and set random seed

        if self._run.tf_strategy is None:
            self._run.tf_strategy = tensorflow.distribute.get_strategy()

        if self.seed is None:
            self.seed = time.time_ns()

        numpy.random.seed(self.seed)
        tensorflow.random.set_seed(self.seed)
        random.seed(self.seed)

        ## Step 3. Generate neural network architecture

        num_outputs = outputs.shape[1]

        output_activation, run_loss = compute_network_configuration(num_outputs, self.dataset)

        delta, widths, network = find_best_layout_for_budget_and_depth(
            inputs,
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

        ## Step 4: Create and execute network using Keras

        with self._run.tf_strategy.scope():

            ### Build Keras model
            self._run.keras_model = build_keras_network(network)
            assert len(self._run.keras_model.inputs) == 1, 'Wrong number of keras inputs generated'
            
            ### Execute Keras model
            self._run.result = self.test_keras_network()

    def log_result(self) -> None:
        engine, session = _connect()
        for record_class in [BaseRecord, ValLossRecord, HistoryRecord]:
            record_element = self.make_dataclass_from_dict(self._run.result, record_class)
            session.add(record_element)
        session.commit()
        # TODO: finish this guy

    def make_dataclass_from_dict(self, source: {}, cls: Type) -> any:
        keys = {f.name for f in dataclasses.fields(BaseRecord) if not f.name.startswith('__')}
        return cls(**{k: v for k, v in source if k in keys})

    def test_keras_network(self) -> None:
        """
        test_keras_network

        Given a fully constructed Keras network, train and test it on a given dataset using hyperparameters in config
        This function also creates log events during and after training.
        """

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

        model = self._run.keras_model

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

        run_callbacks = []
        if self.early_stopping != False:
            run_callbacks.append(callbacks.EarlyStopping(**self.early_stopping))

        run_config = self.run_config.copy()

        if self.test_split > 0:
            ## train/test/val split
            inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs,
                                                                                      test_size=self.test_split])
            run_config["validation_split"] = run_config["validation_split"] / (1 - self.test_split)

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

        ## Checkpoint Code
        if self.checkpoint_epochs > 0:

            assert self.test_split == 0, "Checkpointing is not compatible with test_split."

            DMP_CHECKPOINT_DIR = os.getenv("DMP_CHECKPOINT_DIR", default="checkpoints")
            if not os.path.exists(DMP_CHECKPOINT_DIR):
                os.makedirs(DMP_CHECKPOINT_DIR)

            save_path = os.path.join(DMP_CHECKPOINT_DIR,
                                     self.uuid + ".h5")

            model = ResumableModel(model,
                                   save_every_epochs=self.checkpoint_epochs,
                                   to_path=save_path)

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

        result = self._run.result

        result.num_weights = count_trainable_parameters_in_keras_model(model)
        result.num_inputs = inputs.shape[1]
        result.num_features = dataset['n_features']
        result.num_classes = dataset['n_classes']
        result.num_outputs = outputs.shape[1]
        result.num_observations = inputs.shape[0]
        result.task = dataset['Task']
        result.endpoint = dataset['Endpoint']

        result.history = history

        validation_losses = numpy.array(history['val_loss'])
        best_index = numpy.argmin(validation_losses)

        result.iterations = best_index + 1
        result.val_loss = validation_losses[best_index]
        result.loss = history['loss'][best_index]

        if self.test_split > 0:
            ## Record the history of test evals from the callback
            result.history.update(test_history_callback.history)
            test_losses = numpy.array(history['test_loss'])
            result.test_loss = test_losses[best_index]

        result.environment = {
            "tensorflow_version": tensorflow.__version__,
        }

        #result.run_name = run_name # I think this is deprecated

        return
