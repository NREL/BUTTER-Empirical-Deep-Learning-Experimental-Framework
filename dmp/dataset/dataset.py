from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Sequence
import tensorflow
import sklearn.model_selection
import numpy
from .pmlb import pmlb_loader
from .dataset_spec import DatasetSpec


@dataclass
class Dataset():
    ml_task: str
    input_shape: Sequence[int]
    output_shape: Sequence[int]
    train_data: Any
    validation_data: Any
    test_data: Any

    @staticmethod
    def make(spec: DatasetSpec, batch_size: int) -> 'Dataset':
        (
            dataset_series,
            inputs,
            outputs,
        ) = pmlb_loader.load_dataset(
            pmlb_loader.get_datasets(),
            spec.name,
        )

        ml_task = str(dataset_series['Task'])
        input_shape = inputs.shape
        output_shape = outputs.shape

        (
            train_data,
            validation_data,
            test_data,
        ) = tuple((
            make_tensorflow_dataset(
                dataset,
                batch_size,
                # task.fit_config['batch_size'],
            ) for dataset in split_dataset(
                spec,
                ml_task,
                inputs,
                outputs,
            )))

        return Dataset(
            ml_task,
            input_shape,
            output_shape,
            train_data,
            validation_data,
            test_data,
        )


def split_dataset(
    spec: DatasetSpec,
    run_task: str,
    inputs,
    outputs,
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any], ]:
    validation_inputs, validation_outputs = (None, None)
    train_inputs, train_outputs = (None, None)
    test_inputs, test_outputs = (None, None)

    method = spec.method
    validation_split = spec.validation_split
    if method == 'shuffled_train_test_split':
        train_inputs, test_inputs, train_outputs, test_outputs = \
                sklearn.model_selection.train_test_split(
                    inputs,
                    outputs,
                    test_size=spec.test_split,
                    shuffle=True,
                )

        if validation_split is not None and validation_split > 0.0:
            train_inputs, validation_inputs, train_outputs, validation_outputs = \
                sklearn.model_selection.train_test_split(
                    train_inputs,
                    train_outputs,
                    test_size=int(validation_split/(1-spec.test_split)),
                    shuffle=True,
                )

        add_label_noise(spec.label_noise, run_task, train_outputs)
    else:
        raise NotImplementedError(f'Unknown test_split_method {method}.')

    return (
        (train_inputs, train_outputs),
        (validation_inputs, validation_outputs),
        (test_inputs, test_outputs),
    )


def make_tensorflow_dataset(
    datasets: Sequence,
    batch_size: int,
) -> Any:
    if datasets[0] is None:
        return None

    dataset_options = tensorflow.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = \
        tensorflow.data.experimental.AutoShardPolicy.DATA

    tf_datasets = tuple((tensorflow.data.Dataset.from_tensor_slices(
        dataset).with_options(dataset_options).astype('float32')
                         for dataset in datasets))

    tf_datasets = tensorflow.data.Dataset.from_tensor_slices(tf_datasets)
    tf_datasets = tf_datasets.with_options(dataset_options)
    tf_datasets = tf_datasets.batch(batch_size)
    return tf_datasets


def add_label_noise(label_noise: float, run_task: str, train_outputs: Any):
    if label_noise <= 0.0:
        return

    train_size = len(train_outputs)
    # print(f'run_task {run_task} output shape {outputs.shape}')
    # print(f'sample\n{outputs_train[0:20, :]}')
    if run_task == 'classification':
        num_to_perturb = int(train_size * label_noise)
        noisy_labels_idx = numpy.random.choice(train_size,
                                               size=num_to_perturb,
                                               replace=False)

        num_outputs = train_outputs.shape[1]
        if num_outputs == 1:
            # binary response variable...
            train_outputs[noisy_labels_idx] ^= 1
        else:
            # one-hot response variable...
            rolls = numpy.random.choice(
                numpy.arange(num_outputs - 1) + 1, noisy_labels_idx.size)
            for i, idx in enumerate(noisy_labels_idx):
                train_outputs[noisy_labels_idx] = numpy.roll(
                    train_outputs[noisy_labels_idx], rolls[i])
            # noisy_labels_new_idx = numpy.random.choice(train_size, size=num_to_perturb, replace=True)
            # outputs_train[noisy_labels_idx] = outputs_train[noisy_labels_new_idx]
    elif run_task == 'regression':
        # mean = numpy.mean(outputs, axis=0)
        std_dev = numpy.std(train_outputs, axis=0)
        # print(f'std_dev {std_dev}')
        noise_std = std_dev * label_noise
        for i in range(train_outputs.shape[1]):
            train_outputs[:, i] += numpy.random.normal(
                loc=0, scale=noise_std[i], size=train_outputs[:, i].shape)
    else:
        raise ValueError(
            f'Do not know how to add label noise to dataset task {run_task}.')
