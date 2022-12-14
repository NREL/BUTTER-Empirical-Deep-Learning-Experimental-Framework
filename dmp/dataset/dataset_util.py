import os
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

import tensorflow.keras as keras
from numpy import ndarray
import pandas
from dmp.dataset.dataset_loader import DatasetLoader
from dmp.dataset.functional_pmlb_dataset_loader import FunctionalPMLBDatasetLoader
from dmp.dataset.keras_dataset_loader import KerasDatasetLoader
from dmp.dataset.keras_mnist_dataset_loader import KerasMNISTDatasetLoader
from dmp.dataset.ml_task import MLTask
from dmp.dataset.pmlb_dataset_loader import PMLBDatasetLoader
from dmp.dataset.tf_image_classification_dataset_loader import TFImageClassificationDatasetLoader

from dmp.task.task_util import make_dispatcher


def _make_loader_map(
    loaders: List[DatasetLoader], ) -> Dict[str, DatasetLoader]:
    return {loader.dataset_name: loader for loader in loaders}


_load_keras_dataset = make_dispatcher(
    'keras dataset',
    _make_loader_map([
        KerasMNISTDatasetLoader(
            'mnist',
            keras.datasets.mnist.load_data,
        ),
        KerasMNISTDatasetLoader(
            'fashion_mnist',
            keras.datasets.fashion_mnist.load_data,
        ),
        KerasDatasetLoader(
            'cifar10',
            keras.datasets.cifar10.load_data,
        ),
        KerasDatasetLoader(
            'cifar100',
            lambda: keras.datasets.cifar100.load_data(label_mode='fine'),
        ),
    ]))


def make_load_pmlb_dataset():
    pmlb_index_path = os.path.join(
        os.path.realpath(os.path.join(
            os.getcwd(),
            os.path.dirname(__file__),
        )),
        'pmlb.csv',
    )
    dataset_index = pandas.read_csv(pmlb_index_path)
    dataset_index.set_index('Dataset', inplace=True, drop=False)

    loader_list: List[DatasetLoader] = []
    for _, row in dataset_index.iterrows():
        loader_list.append(PMLBDatasetLoader(row['name'], MLTask(row['task'])))

    loaders: Dict[str, DatasetLoader] = _make_loader_map(loader_list)
    loaders.update(
        _make_loader_map([
            FunctionalPMLBDatasetLoader(
                'mnist',
                MLTask.classification,
                lambda loader, data: loader._prepare_image(data),
            ),
            PMLBDatasetLoader('201_pol', MLTask.classification),
        ]))  # type: ignore
    return make_dispatcher('pmlb dataset', loaders)


_load_pmlb_dataset = make_load_pmlb_dataset()

_source_loaders = make_dispatcher(
    'dataset source', {
        'keras': _load_keras_dataset,
        'tensorflow': TFImageClassificationDatasetLoader,
        'pmlb': _load_pmlb_dataset
    })


def load_dataset(source: str,
                 name: str) -> Tuple[pandas.Series, ndarray, ndarray]:
    return _source_loaders(source)(name)() # type: ignore
