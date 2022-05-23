import numbers
import os
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
)

import numpy
import numpy as np
import pandas
import pmlb
from numpy import ndarray
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)

dataset_path = os.path.join(os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))), 'pmlb.csv')


def load_dataset_index(filePath: str = dataset_path) -> pandas.DataFrame:
    datasets = pandas.read_csv(filePath)
    datasets.set_index('Dataset', inplace=True, drop=False)
    return datasets


def load_dataset(datasets: pandas.DataFrame, dataset_name: str) -> Tuple[pandas.Series, ndarray, ndarray]:
    matching_datasets = datasets[datasets['Dataset'] == dataset_name]
    if len(matching_datasets) <= 0:
        raise Exception('No matching dataset "{}".'.format(dataset_name))
    dataset = matching_datasets.iloc[0].copy()

    # check cache first for raw inputs and outputs in the working directory
    cache_directory = os.path.join(os.getcwd(), ".pmlb_cache", dataset_name)
    os.makedirs(cache_directory, exist_ok=True)
    raw_inputs, raw_outputs = _read_raw_pmlb(cache_directory, dataset_name)

    loader = _default_loader
    if dataset_name in _custom_loaders:
        loader = _custom_loaders[dataset_name]

    inputs, outputs, task = loader(raw_inputs, raw_outputs)
    if task is not None:
        dataset['Task'] = task
    return dataset, inputs, outputs


def _save_raw_pmlb(cache_directory, raw_inputs, raw_outputs):
    with open(os.path.join(cache_directory, 'data.npy'), 'wb') as f:
        numpy.save(f, raw_inputs)
        numpy.save(f, raw_outputs)


def _read_raw_pmlb(cache_directory, dataset_name):
    """ See if the file has been cached and try to read that, download otherwise"""
    try:
        with open(os.path.join(cache_directory, 'data.npy'), 'rb') as f:
            raw_inputs = numpy.load(f, allow_pickle=True)
            raw_outputs = numpy.load(f, allow_pickle=True)
    except FileNotFoundError:
        raw_inputs, raw_outputs = pmlb.fetch_data(dataset_name, return_X_y=True)
        _save_raw_pmlb(cache_directory, raw_inputs, raw_outputs)

    return raw_inputs, raw_outputs


def _default_loader(raw_inputs: ndarray, raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = _prepare_data(raw_inputs)
    outputs = _prepare_data(raw_outputs)
    return inputs, outputs, None


def _load_MNIST(raw_inputs: ndarray, raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = _prepare_matrix(raw_inputs, lambda value: value / 255.0)
    outputs = _prepare_value(raw_outputs, _one_hot)
    return inputs, outputs, 'classification'


def _load_201_pol(raw_inputs: ndarray, raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs, outputs, _ = _default_loader(raw_inputs, raw_outputs)
    return inputs, outputs, 'classification'


_custom_loaders: Dict[str, Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]]] = {
    'mnist': _load_MNIST,
    '201_pol': _load_201_pol,
}


def _prepare_data(value) -> ndarray:
    shape = value.shape
    if len(shape) == 1:
        return _prepare_value(value)
    if len(shape) > 1:
        return _prepare_matrix(value)
    raise Exception('Invalid shape {}.'.format(shape))


def _dynamic_value_transform(value: ndarray) -> Optional[ndarray]:
    # TODO: Normalizer and PCA decorrelation can also help, etc
    # see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    # use one-hot when there are fewer distinct values than 10%
    # of the number of observations

    num_distinct_values = numpy.unique(value).size
    preprocessed = None
    if num_distinct_values <= 1:
        return None
    elif num_distinct_values <= 2:
        preprocessed = _binary(value)
    elif num_distinct_values <= 20 or not isinstance(value[0][0],
                                                     numbers.Number):  # num_distinct_values > .01 * value.shape[0]
        preprocessed = _one_hot(value)
    else:
        preprocessed = _min_max(value)

    return preprocessed


def _prepare_value(
        value: ndarray,
        value_transform: Optional[Callable[[ndarray], ndarray]] = _dynamic_value_transform,
) -> Optional[ndarray]:
    value = numpy.reshape(value, (-1, 1))
    return value_transform(value)


def _prepare_matrix(
        values: ndarray,
        value_transform: Optional[Callable[[ndarray], ndarray]] = _dynamic_value_transform,
) -> ndarray:
    transformed_list = []
    for i in range(values.shape[1]):
        value = values[:, i]
        transformed_value = _prepare_value(value, value_transform)
        if transformed_value is not None:
            transformed_list.append(transformed_value)
    return numpy.hstack(transformed_list)


def _min_max(value: ndarray) -> ndarray:
    preprocessor = MinMaxScaler()
    preprocessor.fit(value)
    return preprocessor.transform(value)


def _one_hot(value: ndarray) -> ndarray:
    preprocessor = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor.fit(value)
    return preprocessor.transform(value)


def _binary(value: ndarray) -> ndarray:
    # if there are only two values, set them as 0 and 1
    return (value == value[0]).astype(np.int)
