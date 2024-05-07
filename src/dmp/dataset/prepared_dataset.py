from typing import List, Dict, Tuple, Optional, Any, Sequence

import sklearn.model_selection
import sklearn.utils
import numpy
from dmp.dataset.dataset import Dataset

from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.ml_task import MLTask
from .dataset_spec import DatasetSpec


class PreparedDataset:
    def __init__(self, spec: DatasetSpec, batch_size: int) -> None:
        from dmp.dataset.dataset_util import load_dataset

        dataset: Dataset = load_dataset(spec.source, spec.name)
        split_dataset(spec, dataset)

        self.ml_task: MLTask = dataset.ml_task
        self.input_shape: List[int] = dataset.input_shape
        self.output_shape: List[int] = dataset.output_shape

        def get_group_size(group) -> int:
            if group is None or group.inputs is None:
                return 0
            return int(group.inputs.shape[0])  # type: ignore

        self.train_size: int = get_group_size(dataset.train)
        self.test_size: int = get_group_size(dataset.test)
        self.validation_size: int = get_group_size(dataset.validation)

        self.train = make_tensorflow_dataset(
            dataset.train,
            batch_size,
        )
        dataset.train = None
        self.test = make_tensorflow_dataset(
            dataset.test,
            batch_size,
        )
        dataset.test = None
        self.validation = make_tensorflow_dataset(
            dataset.validation,
            batch_size,
        )
        dataset.validation = None
        del dataset


def split_dataset(spec: DatasetSpec, dataset: Dataset) -> None:
    method = spec.method
    if method == "shuffled_train_test_split":
        # combine all splits and resplit the dataset

        splits = dataset.splits
        inputs = numpy.concatenate([group.inputs for _, group in splits])
        outputs = numpy.concatenate([group.outputs for _, group in splits])

        del splits
        dataset.train = None
        dataset.test = None
        dataset.validation = None

        (
            train_inputs,
            test_inputs,
            train_outputs,
            test_outputs,
        ) = sklearn.model_selection.train_test_split(
            inputs,
            outputs,
            test_size=spec.test_split,
            shuffle=True,
        )
        dataset.train = DatasetGroup(train_inputs, train_outputs)

        make_validation_split(spec, dataset)
        add_label_noise(spec.label_noise, dataset.ml_task, train_outputs)
        dataset.test = DatasetGroup(test_inputs, test_outputs)
    elif method == "default":
        set_spec_splits_to_actuals(spec, dataset)
    elif method == "default_shuffled":
        for name, group in dataset.splits:
            sklearn.utils.shuffle(group.inputs, group.outputs)
        set_spec_splits_to_actuals(spec, dataset)
    elif method == "default_test":
        make_validation_split(spec, dataset)
    elif method == "swap_test_and_validation":
        temp = dataset.validation
        dataset.validation = dataset.test
        dataset.test = temp
        set_spec_splits_to_actuals(spec, dataset)
    else:
        raise NotImplementedError(f"Unknown test_split_method {method}.")


def set_spec_splits_to_actuals(spec: DatasetSpec, dataset: Dataset):
    total_size = sum([group.inputs.shape[0] for name, group in dataset.splits])

    if dataset.test is None:
        spec.test_split = 0.0
    else:
        spec.test_split = dataset.test.inputs.shape[0] / total_size

    if dataset.validation is None:
        spec.validation_split = 0.0
    else:
        spec.validation_split = dataset.validation.inputs.shape[0] / total_size


def make_validation_split(spec: DatasetSpec, dataset: Dataset) -> None:
    validation_split = spec.validation_split
    if validation_split is None or validation_split <= 0.0:
        spec.validation_split = 0.0
        return

    (
        train_inputs,
        validation_inputs,
        train_outputs,
        validation_outputs,
    ) = sklearn.model_selection.train_test_split(
        dataset.train.inputs,
        dataset.train.outputs,
        test_size=(validation_split / (1.0 - spec.test_split)),
        shuffle=False,
    )
    dataset.train = DatasetGroup(train_inputs, train_outputs)
    dataset.validation = DatasetGroup(validation_inputs, validation_outputs)


def make_tensorflow_dataset(
    group: Optional[DatasetGroup],
    batch_size: int,
) -> Any:
    if group is None:
        return None

    datasets = (group.inputs, group.outputs)

    import tensorflow

    dataset_options = tensorflow.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = (
        tensorflow.data.experimental.AutoShardPolicy.DATA
    )

    tf_datasets = tensorflow.data.Dataset.from_tensor_slices(datasets)

    tf_datasets = tf_datasets.with_options(dataset_options)
    tf_datasets = tf_datasets.batch(batch_size)
    return tf_datasets


def add_label_noise(label_noise: float, ml_task: MLTask, train_outputs: Any):
    if label_noise <= 0.0:
        return

    train_size = len(train_outputs)
    # print(f'run_task {run_task} output shape {outputs.shape}')
    # print(f'sample\n{outputs_train[0:20, :]}')
    if ml_task == MLTask.classification:
        num_to_perturb = int(train_size * label_noise)
        noisy_labels_idx = numpy.random.choice(
            train_size, size=num_to_perturb, replace=False
        )

        num_outputs = train_outputs.shape[1]
        if num_outputs == 1:
            # binary response variable...
            train_outputs[noisy_labels_idx] ^= 1
        else:
            # one-hot response variable...
            rolls = numpy.random.choice(
                numpy.arange(num_outputs - 1) + 1, noisy_labels_idx.size
            )
            for i, idx in enumerate(noisy_labels_idx):
                train_outputs[noisy_labels_idx] = numpy.roll(
                    train_outputs[noisy_labels_idx], rolls[i]
                )
            # noisy_labels_new_idx = numpy.random.choice(train_size, size=num_to_perturb, replace=True)
            # outputs_train[noisy_labels_idx] = outputs_train[noisy_labels_new_idx]
    elif ml_task == MLTask.regression:
        # mean = numpy.mean(outputs, axis=0)
        std_dev = numpy.std(train_outputs, axis=0)
        # print(f'std_dev {std_dev}')
        noise_std = std_dev * label_noise
        for i in range(train_outputs.shape[1]):
            train_outputs[:, i] += numpy.random.normal(
                loc=0, scale=noise_std[i], size=train_outputs[:, i].shape
            )
    else:
        raise ValueError(
            f"Do not know how to add label noise to dataset task {ml_task}."
        )
