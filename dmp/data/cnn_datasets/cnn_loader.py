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
from numpy import ndarray
import tensorflow.keras.datasets as keras_datasets
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)

'''
CNN Datasets and Where to Find Them
MNIST: load from keras
CIFAR-10: load from keras
CIFAR-100: load from keras
Fashion-MNIST: load from keras

'''

dataset_path = os.path.join(os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))), 'cnn_datasets.csv')


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
    cache_directory = os.path.join(os.getcwd(), ".cnn_dataset_cache", dataset_name)
    os.makedirs(cache_directory, exist_ok=True)
    raw_inputs, raw_outputs = _read_raw_cnn_data(cache_directory, dataset_name)

    loader = _default_loader
    if dataset_name in _custom_loaders:
        loader = _custom_loaders[dataset_name]

    inputs, outputs, task = loader(raw_inputs, raw_outputs)
    if task is not None:
        dataset['Task'] = task
    return dataset, inputs, outputs


def _save_raw_cnn_data(cache_directory, raw_inputs, raw_outputs):
    with open(os.path.join(cache_directory, 'data.npy'), 'wb') as f:
        numpy.save(f, raw_inputs)
        numpy.save(f, raw_outputs)

def _fetch_keras_data(dataset_name: str) -> Tuple[ndarray, ndarray]:
    if dataset_name == 'mnist':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.cifar10.load_data()
    elif dataset_name == 'cifar100':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.cifar100.load_data(label_mode='fine')
    else:
        raise Exception('No matching dataset "{}".'.format(dataset_name))
    # concatenate train and test data into raw_inputs, raw_outputs
    raw_inputs = np.concatenate((xtrain, xtest), axis=0)
    raw_outputs = np.concatenate((ytrain, ytest), axis=0)
    return raw_inputs, raw_outputs

def _fetch_tf_data(dataset_name: str) -> Tuple[ndarray, ndarray]:
    import tensorflow_datasets as tfds
    dl_config = tfds.download.DownloadConfig(verify_ssl=False)
    if dataset_name == 'colorectal_histology':
        ds = tfds.load('dataset_name', 
                        split='train', 
                        shuffle_files=False,
                        as_supervised=True, 
                        download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'eurosat_all':
        ds = tfds.load('eurosat/all', 
                        split='train', 
                        shuffle_files=False,
                        as_supervised=True, 
                        download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'eurosat_rgb':
        ds = tfds.load('eurosat/rgb', 
                        split='train', 
                        shuffle_files=False,
                        as_supervised=True, 
                        download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'horses_or_humans':
        ds = tfds.load('horses_or_humans', 
                        split='train', 
                        shuffle_files=False,
                        as_supervised=True, 
                        download_and_prepare_kwargs={'download_config': dl_config})
        ds1 = tfds.load('horses_or_humans',
                        split='test',
                        shuffle_files=False,
                        as_supervised=True,
                        download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'patch_camelyon' or dataset_name == 'places365_small':
        ds = tfds.load(dataset_name, 
                        split='train', 
                        shuffle_files=False,
                        as_supervised=True, 
                        download_and_prepare_kwargs={'download_config': dl_config})
        ds1 = tfds.load(dataset_name,
                        split='test',
                        shuffle_files=False,
                        as_supervised=True,
                        download_and_prepare_kwargs={'download_config': dl_config})
        ds2 = tfds.load(dataset_name,
                        split='validation',
                        shuffle_files=False,
                        as_supervised=True,
                        download_and_prepare_kwargs={'download_config': dl_config})
    else:
        raise Exception('No matching dataset "{}".'.format(dataset_name))
    ds = tfds.as_numpy(ds)
    raw_inputs = []
    raw_outputs = []
    for ex in ds:
        raw_inputs.append(ex[0])
        raw_outputs.append(ex[1])
    if dataset_name == 'horses_or_humans' or dataset_name == 'patch_camelyon' or dataset_name == 'places365_small':
        ds1 = tfds.as_numpy(ds1)
        for ex in ds1:
            raw_inputs.append(ex[0])
            raw_outputs.append(ex[1])
        if dataset_name == 'patch_camelyon' or dataset_name == 'places365_small':
            ds2 = tfds.as_numpy(ds2)
            for ex in ds2:
                raw_inputs.append(ex[0])
                raw_outputs.append(ex[1])
    raw_inputs = np.array(raw_inputs)
    raw_outputs = np.array(raw_outputs)
    return raw_inputs, raw_outputs

def _fetch_data(dataset_name: str) -> Tuple[ndarray, ndarray]:
    keras_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
    tf_datasets = ['colorectal_histology', 'eurosat_all', 'eurosat_rgb', 
                'horses_or_humans', 'patch_camelyon', 'places365_small']
    if dataset_name in keras_datasets:
        raw_inputs, raw_outputs = _fetch_keras_data(dataset_name)
    elif dataset_name in tf_datasets:
        raw_inputs, raw_outputs = _fetch_tf_data(dataset_name)
    else:
        raise Exception('No matching dataset "{}".'.format(dataset_name))
    return raw_inputs, raw_outputs

def _read_raw_cnn_data(cache_directory, dataset_name):
    """ See if the file has been cached and try to read that, download otherwise"""
    try:
        with open(os.path.join(cache_directory, 'data.npy'), 'rb') as f:
            raw_inputs = numpy.load(f, allow_pickle=True)
            raw_outputs = numpy.load(f, allow_pickle=True)
    except FileNotFoundError:
        raw_inputs, raw_outputs = _fetch_data(dataset_name)
        _save_raw_cnn_data(cache_directory, raw_inputs, raw_outputs)

    return raw_inputs, raw_outputs


def _default_loader(raw_inputs: ndarray, raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = _prepare_data(raw_inputs)
    outputs = _prepare_data(raw_outputs)
    return inputs, outputs, None


def _load_image(raw_inputs: ndarray, raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = raw_inputs / 255.0
    outputs = _prepare_value(raw_outputs, _one_hot)
    return inputs, outputs, 'classification'

def _load_image_binary(raw_inputs: ndarray, raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = raw_inputs / 255.0
    outputs = _prepare_value(raw_outputs, _binary)
    return inputs, outputs, 'classification'

_custom_loaders: Dict[str, Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]]] = {
    'mnist': _load_image,
    'fashion_mnist': _load_image,
    'cifar10': _load_image,
    'cifar100': _load_image,
    'colorectal_histology': _load_image,
    'eurosat_all': _load_image,
    'eurosat_rgb': _load_image,
    'places365_small': _load_image,
    'horses_or_humans': _load_image_binary,
    'patch_camelyon': _load_image_binary,
}


def _prepare_data(value) -> ndarray:
    shape = value.shape
    if len(shape) == 1:
        return _prepare_value(value)
    elif len(shape) > 1:
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

def _prepare_tensor(
        values: ndarray,
        value_transform: Optional[Callable[[ndarray], ndarray]] = _dynamic_value_transform,
) -> ndarray:
    # apply value_transform to all entries
    return value_transform(values)


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