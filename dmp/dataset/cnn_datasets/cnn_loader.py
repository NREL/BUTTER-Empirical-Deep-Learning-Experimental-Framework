import numbers
import os
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
)
from xmlrpc.client import _binary

import numpy
import numpy as np
import pandas
from numpy import ndarray
import tensorflow.keras.datasets as keras_datasets
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)

import pickle
import tempfile

from dmp.dataset.dataset_util import default_loader, one_hot, prepare_data, prepare_value, binary

# from  dmp.dataset.dataset_util import
'''
CNN Datasets and Where to Find Them
MNIST: load from keras
CIFAR-10: load from keras
CIFAR-100: load from keras
Fashion-MNIST: load from keras

'''

dataset_path = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
    'cnn_datasets.csv')
imagenet_path = os.path.join(os.getcwd(), "image_net")


def load_dataset(
    datasets: pandas.DataFrame,
    dataset_name: str,
    cache_directory : str,
) -> Tuple[pandas.Series, ndarray, ndarray]:
    matching_datasets = datasets[datasets['Dataset'] == dataset_name]
    if len(matching_datasets) <= 0:
        raise Exception('No matching dataset "{}".'.format(dataset_name))
    dataset = matching_datasets.iloc[0].copy()

    # check cache first for raw inputs and outputs in the working directory
    cache_directory = os.path.join(os.getcwd(), '.dataset_cache', dataset_name)
    os.makedirs(cache_directory, exist_ok=True)
    raw_inputs, raw_outputs = _read_raw_cnn_data(cache_directory, dataset_name)

    loader = default_loader
    if dataset_name in _custom_loaders:
        loader = _custom_loaders[dataset_name]

    inputs, outputs, task = loader(raw_inputs, raw_outputs)
    if task is not None:
        dataset['Task'] = task
    return dataset, inputs, outputs


def _fetch_keras_data(dataset_name: str) -> Tuple[ndarray, ndarray]:
    if dataset_name == 'mnist_keras':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (xtrain, ytrain), (xtest,
                           ytest) = keras_datasets.fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.cifar10.load_data()
    elif dataset_name == 'cifar100':
        (xtrain, ytrain), (xtest, ytest) = keras_datasets.cifar100.load_data(
            label_mode='fine')
    else:
        raise Exception('No matching dataset "{}".'.format(dataset_name))
    # concatenate train and test data into raw_inputs, raw_outputs
    raw_inputs = np.concatenate((xtrain, xtest), axis=0)
    if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
        raw_inputs = raw_inputs.reshape(raw_inputs.shape[0], 28, 28, 1)
    raw_outputs = np.concatenate((ytrain, ytest), axis=0)
    return raw_inputs, raw_outputs


def _fetch_tf_data(dataset_name: str) -> Tuple[ndarray, ndarray]:
    import tensorflow_datasets as tfds
    dl_config = tfds.download.DownloadConfig(verify_ssl=False)
    if dataset_name == 'colorectal_histology':
        ds = tfds.load(
            'dataset_name',
            split='train',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'eurosat_all':
        ds = tfds.load(
            'eurosat/all',
            split='train',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'eurosat_rgb':
        ds = tfds.load(
            'eurosat/rgb',
            split='train',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'horses_or_humans':
        ds = tfds.load(
            'horses_or_humans',
            split='train',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
        ds1 = tfds.load(
            'horses_or_humans',
            split='test',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
    elif dataset_name == 'patch_camelyon' or dataset_name == 'places365_small':
        ds = tfds.load(
            dataset_name,
            split='train',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
        ds1 = tfds.load(
            dataset_name,
            split='test',
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})
        ds2 = tfds.load(
            dataset_name,
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


def _fetch_imagenet_data(size: int = 16,
                         crop_120: bool = False) -> Tuple[ndarray, ndarray]:
    datafolder = imagenet_path
    if size == 16:
        train_path = datafolder + 'Imagenet32_train_npz/train_data_batch_'
        test_path = datafolder + 'Imagenet32_val_npz/val_data.npz'
        s = 16
    elif size == 32:
        train_path = datafolder + 'Imagenet32_train_npz/train_data_batch_'
        test_path = datafolder + 'Imagenet32_val_npz/val_data.npz'
        s = 32
    else:
        raise Exception('No matching dataset "imagenet_{}".'.format(size))

    batches = 10
    d = np.load(test_path)
    # print('loaded test data')
    raw_inputs = d['data']
    raw_outputs = d['labels']
    for i in range(1, batches + 1):
        path = train_path + str(i) + '.npz'
        d = np.load(path)
        raw_inputs = np.concatenate((raw_inputs, d['data']), axis=0)
        raw_outputs = np.concatenate((raw_outputs, d['labels']), axis=0)
    raw_outputs -= 1  # subtract 1 to make labels start at 0
    if crop_120:
        inds = raw_outputs < 120
        raw_inputs = raw_inputs[inds]
        raw_outputs = raw_outputs[inds]
    raw_outputs = raw_outputs.astype(int)
    n = raw_outputs.shape[0]
    raw_inputs = raw_inputs.reshape(n, 3, s, s)
    raw_inputs = np.transpose(raw_inputs, (0, 2, 3, 1))

    return raw_inputs, raw_outputs


def _fetch_data(dataset_name: str) -> Tuple[ndarray, ndarray]:
    keras_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
    tf_datasets = [
        'colorectal_histology', 'eurosat_all', 'eurosat_rgb',
        'horses_or_humans', 'patch_camelyon', 'places365_small'
    ]
    if dataset_name in keras_datasets:
        raw_inputs, raw_outputs = _fetch_keras_data(dataset_name)
    elif dataset_name in tf_datasets:
        raw_inputs, raw_outputs = _fetch_tf_data(dataset_name)
    elif dataset_name == 'imagenet_16':
        raw_inputs, raw_outputs = _fetch_imagenet_data(size=16, crop_120=False)
    elif dataset_name == 'imagenet_16_120':
        raw_inputs, raw_outputs = _fetch_imagenet_data(size=16, crop_120=True)
    elif dataset_name == 'imagenet_32':
        raw_inputs, raw_outputs = _fetch_imagenet_data(size=32, crop_120=False)
    elif dataset_name == 'imagenet_32_120':
        raw_inputs, raw_outputs = _fetch_imagenet_data(size=32, crop_120=True)
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


def _load_image(
        raw_inputs: ndarray,
        raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = raw_inputs / 255.0
    outputs = prepare_value(raw_outputs, one_hot)
    return inputs, outputs, 'classification'


def _load_image_binary(
        raw_inputs: ndarray,
        raw_outputs: ndarray) -> Tuple[ndarray, ndarray, Optional[str]]:
    inputs = raw_inputs / 255.0
    outputs = prepare_value(raw_outputs, binary)
    return inputs, outputs, 'classification'


_custom_loaders: Dict[str,
                      Callable[[ndarray, ndarray],
                               Tuple[ndarray, ndarray, Optional[str]]]] = {
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
                                   'imagenet_16_120': _load_image,
                                   'imagenet_16': _load_image,
                                   'imagenet_32_120': _load_image,
                                   'imagenet_32': _load_image,
                               }
