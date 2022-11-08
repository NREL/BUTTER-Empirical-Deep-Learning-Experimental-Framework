import json
from dataclasses import dataclass
from typing import Any

import numpy
import pandas
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.metrics as metrics
from dmp.data.pmlb import pmlb_loader
from dmp.jobqueue_interface.common import jobqueue_marshal
from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from dmp.task.aspect_test.aspect_test_utils import *
from dmp.task.task import Parameter
from dmp.task.task_util import remap_key_prefixes
from pytest import param

# from keras_buoy.models import ResumableModel


@dataclass
class AspectTestExecutor():
    '''
    '''
    task: AspectTestTask
    output_activation: Optional[str] = None
    # tensorflow_strategy: Optional[tensorflow.distribute.Strategy] = None
    keras_model: Optional[tensorflow.keras.Model] = None
    run_loss: Optional[tensorflow.keras.losses.Loss] = None
    network_structure: Optional[NetworkModule] = None
    dataset_series: Optional[pandas.Series] = None
    inputs: Optional[numpy.ndarray] = None
    outputs: Optional[numpy.ndarray] = None
    widths: Optional[List[int]] = None

    def load_dataset(self):
        # Load dataset
        self.dataset_series, self.inputs, self.outputs =  \
            pmlb_loader.load_dataset(
                pmlb_loader.get_datasets(),
                self.task.dataset)

    def make_network(self):
        task = self.task

        # Generate neural network architecture
        num_outputs = self.outputs.shape[1]
        self.output_activation, self.run_loss = \
            compute_network_configuration(num_outputs, self.dataset_series)

        # TODO: make it so we don't need this hack
        shape = task.shape
        residual_mode = None
        residual_suffix = '_residual'
        if shape.endswith(residual_suffix):
            residual_mode = 'full'
            shape = shape[0:-len(residual_suffix)]

        layer_args = {
            'kernel_regularizer': task.kernel_regularizer,
            'bias_regularizer': task.bias_regularizer,
            'activity_regularizer': task.activity_regularizer,
        }

        # Build NetworkModule network
        delta, self.widths, self.network_structure = \
            find_best_layout_for_budget_and_depth(
                self.inputs.shape,
                residual_mode,
                task.input_activation,
                task.activation,
                self.output_activation,
                task.size,
                widths_factory(shape)(num_outputs, task.depth),
                layer_args,
            )

        # reject non-conformant network sizes
        delta = count_num_free_parameters(self.network_structure) - task.size
        relative_error = delta / task.size
        if numpy.abs(relative_error) > .2:
            raise ValueError(
                f'Could not find conformant network error : {relative_error}%, delta : {delta}, size: {self.size}.')

    def make_keras_model(self) -> list:
        task = self.task

        with worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)
            print(f'Tensorflow scope: {s}')
            # Build Keras model
            self.keras_model = make_keras_network_from_network_module(
                self.network_structure)
            if len(self.keras_model.inputs) != 1:  # type: ignore
                print(
                    f'weird error: {len(self.keras_model.inputs)}, {json.dumps(jobqueue_marshal.marshal(self.network_structure))}')  # type: ignore
                raise ValueError('Wrong number of keras inputs generated')

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

            run_optimizer = tensorflow.keras.optimizers.get(task.optimizer)

        self.keras_model.compile(
            loss=self.run_loss,
            optimizer=run_optimizer,
            metrics=run_metrics,
            run_eagerly=False,
        )

        # Calculate number of parameters in grown network
        num_free_parameters = count_trainable_parameters_in_keras_model(
            self.keras_model)
        if count_num_free_parameters(self.network_structure) \
                != num_free_parameters:
            raise RuntimeError('Wrong number of trainable parameters')

        # Configure Keras Callbacks
        callbacks = []
        if task.early_stopping is not None:
            callbacks.append(
                callbacks.EarlyStopping(**task.early_stopping))

        # # optionally enable checkpoints
        # if self.save_every_epochs is not None and self.save_every_epochs > 0:
        #     DMP_CHECKPOINT_DIR = os.getenv(
        #         'DMP_CHECKPOINT_DIR', default='checkpoints')
        #     if not os.path.exists(DMP_CHECKPOINT_DIR):
        #         os.makedirs(DMP_CHECKPOINT_DIR)

        #     save_path = os.path.join(
        #         DMP_CHECKPOINT_DIR, self.job_id + '.h5')

        #     self.keras_model = ResumableModel(
        #         self.keras_model,
        #         save_every_epochs=self.save_every_epochs,
        #         to_path=save_path)
        return callbacks

    def make_tensorflow_dataset_converter(
        self,
        prepared_config: dict,
    ):
        dataset_options = tensorflow.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = \
            tensorflow.data.experimental.AutoShardPolicy.DATA

        inputs_dataset = tensorflow.data.Dataset.from_tensor_slices(
            self.inputs)
        inputs_dataset = inputs_dataset.with_options(dataset_options)

        outputs_dataset = tensorflow.data.Dataset.from_tensor_slices(
            self.outputs)
        outputs_dataset = outputs_dataset.with_options(dataset_options)

        def make_tensorflow_dataset(x, y):
            print(f' make_tf_ds {x.shape}, {y.shape}')
            x = x.astype('float32')
            y = y.astype('float32')
            ds = tensorflow.data.Dataset.from_tensor_slices((x, y))
            ds = ds.with_options(dataset_options)
            ds = ds.batch(task.run_config['batch_size'])
            print(f'ds inspection: {ds.element_spec}')
            return ds
        return make_tensorflow_dataset

    def setup_tensorflow_training_and_validation_datasets(
        self,
        prepared_config,
        make_tensorflow_dataset,
    ) -> None:
        prepared_config['x'] = make_tensorflow_dataset(
            prepared_config['x'], prepared_config['y'])
        del prepared_config['y']

        test_data = prepared_config['validation_data']
        prepared_config['validation_data'] = make_tensorflow_dataset(
            *test_data)

    def fit_model(self,
                  prepared_config: dict,
                  callbacks: list,
                  ) -> Any:
        history = self.keras_model.fit(
            callbacks=callbacks,
            **prepared_config,
        )

        # Tensorflow models return a History object from their fit function,
        # but ResumableModel objects returns History.history. This smooths
        # out that incompatibility.
        if self.task.save_every_epochs is None \
                or self.task.save_every_epochs == 0:
            history = history.history  # type: ignore

        # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
        # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
        # model.save_weights(f'./log/weights/{run_name}.h5', save_format='h5')
        # model.save(f'./log/models/{run_name}.h5', save_format='h5')

        return history

    def __call__(self, parent: AspectTestTask, worker, *args, **kwargs) \
            -> Dict[str, Any]:

        task: AspectTestTask = self.task  # for easy access

        self.seed = set_random_seeds(self.seed)
        self.load_dataset()

        # -----------
        # prepare dataset shuffle, split, and label noise:
        prepared_config = prepare_dataset(
            task.test_split_method,
            task.test_split,
            task.label_noise,
            task.run_config,
            str(self.dataset_series['Task']),
            self.inputs,
            self.outputs,
        )

        # -----------
        self.make_network()
        callbacks = self.make_keras_model()
        make_tensorflow_dataset = \
            self.make_tensorflow_dataset_converter(prepared_config)
        self.setup_tensorflow_training_and_validation_datasets(
            make_tensorflow_dataset,
            prepared_config,
        )

        history = self.fit_model(prepared_config, callbacks)

        # rename 'val_' keys to 'test_' and un-prefixed history keys to 'train_'
        history = remap_key_prefixes(
            history, [
                ('val_', 'test_'),
                ('', 'train_'),
            ])  # type: ignore

        parameters: Dict[str, Any] = parent.parameters
        parameters['output_activation'] = self.output_activation
        parameters['widths'] = self.widths
        parameters['num_free_parameters'] = num_free_parameters
        parameters['output_activation'] = self.output_activation
        parameters['network_structure'] = \
            jobqueue_marshal.marshal(self.network_structure)

        parameters.update(history)
        parameters.update(worker.worker_info)

        # return the result record
        return parameters
