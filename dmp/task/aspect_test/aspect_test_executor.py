from dataclasses import dataclass
from typing import Any

import numpy
import tensorflow.keras.metrics as metrics
from dmp.data.pmlb import pmlb_loader
from dmp.jobqueue_interface.common import jobqueue_marshal
from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from dmp.task.aspect_test.aspect_test_utils import *
from dmp.task.task import Parameter
from dmp.task.task_util import remap_key_prefixes

# from keras_buoy.models import ResumableModel


@dataclass
class AspectTestExecutor():
    '''
    '''

    # task: AspectTestTask
    # output_activation: Optional[str] = None
    # keras_model: Optional[tensorflow.keras.Model] = None
    # run_loss: Optional[tensorflow.keras.losses.Loss] = None
    # network_structure: Optional[NetworkModule] = None

    # dataset_series: Optional[pandas.Series] = None
    # inputs: Optional[numpy.ndarray] = None
    # outputs: Optional[numpy.ndarray] = None

    # widths: Optional[List[int]] = None

    def __call__(self, task: AspectTestTask, worker, *args, **kwargs) \
            -> Dict[str, Any]:

        self.set_random_seeds(task)

        (
            ml_task,
            input_shape,
            output_shape,
            prepared_config,
            make_tensorflow_dataset,
        ) = self.load_and_prepare_dataset(task)

        (
            network_structure,
            widths,
            num_free_parameters,
            run_loss,
            output_activation,
        ) = self.make_network(
            task,
            input_shape,
            output_shape,
            ml_task,
            task.size,
        )


        keras_model, node_layer_map, run_metrics, run_optimizer = \
            self.make_keras_model(task, network_structure)

        self.compile_keras_network(
            network_structure,
            run_loss,
            keras_model,
            run_metrics,
            run_optimizer,
        )

        # Configure Keras Callbacks
        callbacks = []
        if task.early_stopping is not None:
            callbacks.append(
                tensorflow.keras.callbacks.EarlyStopping(
                    **task.early_stopping)) # TODO: decide what to do with this guy

        history = self.fit_model(
            task,
            keras_model,
            prepared_config,
            callbacks,
        )

        # rename 'val_' keys to 'test_' and un-prefixed history keys to 'train_'
        history = remap_key_prefixes(history, [
            ('val_', 'test_'),
            ('', 'train_'),
        ])  # type: ignore

        return self.make_result_record(
            task,
            worker,
            network_structure,
            widths,
            num_free_parameters,
            output_activation,
            history,
        )

    def make_result_record(
        self,
        task: AspectTestTask,
        worker,
        network_structure: NetworkModule,
        widths: List[int],
        num_free_parameters: int,
        output_activation: str,
        history: Dict[str, Any],
    ) -> Dict[str, Any]:
        parameters: Dict[str, Any] = task.parameters
        parameters['widths'] = widths
        parameters['num_free_parameters'] = num_free_parameters
        parameters['output_activation'] = output_activation
        parameters['network_structure'] = \
            jobqueue_marshal.marshal(network_structure)

        parameters.update(history)
        parameters.update(worker.worker_info)

        # return the result record
        return parameters

    def load_and_prepare_dataset(
        self,
        task: AspectTestTask,
        val_portion: Optional[float] = None,
    ) -> Tuple[str, Tuple[int], Tuple[int], dict, Callable]:
        dataset_series, inputs, outputs = self.load_dataset(task)
        ml_task = str(dataset_series['Task'])
        input_shape = inputs.shape
        output_shape = outputs.shape

        # prepare dataset shuffle, split, and label noise:
        prepared_config = prepare_dataset(
            task.test_split_method,
            task.test_split,
            task.label_noise,
            task.run_config,
            ml_task,
            inputs,
            outputs,
            val_portion=val_portion,
        )
        make_tensorflow_dataset = \
            self.make_tensorflow_dataset_converter(
                task,
                inputs,
                outputs,
                prepared_config,
            )
        self.setup_tensorflow_training_and_validation_datasets(
            prepared_config,
            make_tensorflow_dataset,
        )

        return (
            ml_task,
            input_shape,
            output_shape,
            prepared_config,
            make_tensorflow_dataset,
        )

    def set_random_seeds(
        self,
        task: AspectTestTask,
    ) -> None:
        seed: int = task.seed
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)
        random.seed(seed)

    def load_dataset(
        self,
        task: AspectTestTask,
    ) -> Tuple[Any, Any, Any]:
        dataset_series, inputs, outputs =  \
            pmlb_loader.load_dataset(
                pmlb_loader.get_datasets(),
                task.dataset)
        return dataset_series, inputs, outputs

    def make_network(
        self,
        task: AspectTestTask,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        ml_task: str,
        target_size: int,
    ) -> Tuple[NetworkModule, List[int], int, tensorflow.keras.losses.Loss,
               str, ]:
        # Generate neural network architecture
        num_outputs = output_shape[1]
        output_activation, run_loss = \
            compute_network_configuration(num_outputs, ml_task)

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
        delta, widths, network_structure = \
            find_best_layout_for_budget_and_depth(
                input_shape,
                residual_mode,
                task.input_activation,
                task.activation,
                output_activation,
                target_size,
                widths_factory(shape)(num_outputs, task.depth),
                layer_args,
            )

        num_free_parameters = count_num_free_parameters(network_structure)
        # reject non-conformant network sizes
        delta = num_free_parameters - task.size
        relative_error = delta / task.size
        if numpy.abs(relative_error) > .2:
            raise ValueError(
                f'Could not find conformant network error : {relative_error}%, delta : {delta}, size: {self.size}.'
            )

        return (
            network_structure,
            widths,
            num_free_parameters,
            run_loss,
            output_activation,
        )

    def compile_keras_network(self, network_structure, run_loss, keras_model,
                              run_metrics, run_optimizer):
        keras_model.compile(
            loss=run_loss,
            optimizer=run_optimizer,
            metrics=run_metrics,
            run_eagerly=False,
        )

        # Calculate number of parameters in grown network
        num_free_parameters = count_trainable_parameters_in_keras_model(
            keras_model)
        if count_num_free_parameters(network_structure) \
                != num_free_parameters:
            raise RuntimeError('Wrong number of trainable parameters')

        # # optionally enable checkpoints
        # if self.save_every_epochs is not None and self.save_every_epochs > 0:
        #     DMP_CHECKPOINT_DIR = os.getenv(
        #         'DMP_CHECKPOINT_DIR', default='checkpoints')
        #     if not os.path.exists(DMP_CHECKPOINT_DIR):
        #         os.makedirs(DMP_CHECKPOINT_DIR)

        #     save_path = os.path.join(
        #         DMP_CHECKPOINT_DIR, self.job_id + '.h5')

        #     keras_model = ResumableModel(
        #         keras_model,
        #         save_every_epochs=self.save_every_epochs,
        #         to_path=save_path)

    def make_keras_model(self, task, network_structure):
        with worker.strategy.scope() as s:  # type: ignore
            tensorflow.config.optimizer.set_jit(True)

            # Build Keras model
            keras_model, node_layer_map = make_keras_network_from_network_module(
                network_structure)
            if len(keras_model.inputs) != 1:  # type: ignore
                raise ValueError('Wrong number of keras inputs generated')

            # Compile Keras Model
            run_metrics = [
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
        return keras_model, node_layer_map, run_metrics, run_optimizer

    def make_tensorflow_dataset_converter(
        self,
        task: AspectTestTask,
        inputs,
        outputs,
        prepared_config: dict,
    ) -> Callable:
        dataset_options = tensorflow.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = \
            tensorflow.data.experimental.AutoShardPolicy.DATA

        inputs_dataset = tensorflow.data.Dataset.from_tensor_slices(inputs)
        inputs_dataset = inputs_dataset.with_options(dataset_options)

        outputs_dataset = tensorflow.data.Dataset.from_tensor_slices(outputs)
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
        prepared_config: Dict[str, Any],
        make_tensorflow_dataset: Callable,
    ) -> None:
        prepared_config['x'] = make_tensorflow_dataset(prepared_config['x'],
                                                       prepared_config['y'])
        del prepared_config['y']

        test_data = prepared_config['validation_data']
        prepared_config['validation_data'] = make_tensorflow_dataset(
            *test_data)

    def fit_model(
        self,
        task: AspectTestTask,
        keras_model: tensorflow.keras.Model,
        prepared_config: Dict[str, Any],
        callbacks: list,
    ) -> Any:
        history = keras_model.fit(
            callbacks=callbacks,
            **prepared_config,
        )

        # Tensorflow models return a History object from their fit function,
        # but ResumableModel objects returns History.history. This smooths
        # out that incompatibility.
        if task.save_every_epochs is None \
                or task.save_every_epochs == 0:
            history = history.history  # type: ignore

        # Direct method of saving the model (or just weights). This is automatically done by the ResumableModel interface if you enable checkpointing.
        # Using the older H5 format because it's one single file instead of multiple files, and this should be easier on Lustre.
        # model.save_weights(f'./log/weights/{run_name}.h5', save_format='h5')
        # model.save(f'./log/models/{run_name}.h5', save_format='h5')

        return history
