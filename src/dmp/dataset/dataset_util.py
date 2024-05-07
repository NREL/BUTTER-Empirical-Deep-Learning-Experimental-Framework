import os
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from dmp.dataset.gaussian_classification_dataset import GaussianClassificationDataset
import tensorflow.keras as keras
import pandas
from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_loader import DatasetLoader
from dmp.dataset.functional_pmlb_dataset_loader import FunctionalPMLBDatasetLoader
from dmp.dataset.gaussian_regression_dataset import GaussianRegressionDataset
from dmp.dataset.keras_image_dataset_loader import KerasImageDatasetLoader
from dmp.dataset.keras_mnist_dataset_loader import KerasMNISTDatasetLoader
from dmp.dataset.ml_task import MLTask
from dmp.dataset.pmlb_dataset_loader import PMLBDatasetLoader
from dmp.dataset.tf_image_classification_dataset_loader import (
    TFImageClassificationDatasetLoader,
)
from dmp.dataset.imagenet_dataset_loader import ImageNetDatasetLoader
from dmp.common import make_dispatcher


def load_dataset(source: str, name: str) -> Dataset:
    result = __source_loaders(source)(name)()  # type: ignore
    return result


def _make_loader_map(
    loaders: List[DatasetLoader],
) -> Dict[str, DatasetLoader]:
    return {loader.dataset_name: loader for loader in loaders}


__load_keras_dataset = make_dispatcher(
    "keras dataset",
    _make_loader_map(
        [
            KerasMNISTDatasetLoader(
                "mnist",
                keras.datasets.mnist.load_data,
            ),
            KerasMNISTDatasetLoader(
                "fashion_mnist",
                keras.datasets.fashion_mnist.load_data,
            ),
            KerasImageDatasetLoader(
                "cifar10",
                keras.datasets.cifar10.load_data,
            ),
            KerasImageDatasetLoader(
                "cifar100",
                lambda: keras.datasets.cifar100.load_data(label_mode="fine"),
            ),
        ]
    ),
)


def __make_load_pmlb_dataset():
    pmlb_index_path = os.path.join(
        os.path.realpath(
            os.path.join(
                os.getcwd(),
                os.path.dirname(__file__),
            )
        ),
        "pmlb.csv",
    )
    dataset_index = pandas.read_csv(pmlb_index_path)
    dataset_index.set_index("Dataset", inplace=True, drop=False)

    loader_list: List[DatasetLoader] = []
    for _, row in dataset_index.iterrows():
        loader_list.append(PMLBDatasetLoader(row["Dataset"], MLTask(row["Task"])))

    loaders: Dict[str, DatasetLoader] = _make_loader_map(loader_list)
    loaders.update(
        _make_loader_map(
            [
                FunctionalPMLBDatasetLoader(
                    "mnist",
                    MLTask.classification,
                    lambda loader, data: loader._prepare_image(data),
                ),
                PMLBDatasetLoader("201_pol", MLTask.classification),
                PMLBDatasetLoader("294_satellite_image", MLTask.classification),
            ]
        )
    )  # type: ignore
    return make_dispatcher("pmlb dataset", loaders)


__load_pmlb_dataset = __make_load_pmlb_dataset()

__load_imagenet_dataset = make_dispatcher(
    "ImageNet dataset",
    _make_loader_map(
        [
            ImageNetDatasetLoader(
                "imagenet_16",
                MLTask.classification,
                16,
                None,
            ),
            ImageNetDatasetLoader(
                "imagenet_16_120",
                MLTask.classification,
                16,
                120,
            ),
            ImageNetDatasetLoader(
                "imagenet_32",
                MLTask.classification,
                32,
                None,
            ),
            ImageNetDatasetLoader(
                "imagenet_32_120",
                MLTask.classification,
                32,
                120,
            ),
        ]
    ),
)

__load_synthetic_dataset = make_dispatcher(
    "synthetic dataset",
    _make_loader_map(
        [
            GaussianClassificationDataset(2, 10, 1.0, 10000),
            GaussianRegressionDataset(20, 1.0, 1000),
        ]
    ),
)

__source_loaders = make_dispatcher(
    "dataset source",
    {
        "keras": __load_keras_dataset,
        "tensorflow": TFImageClassificationDatasetLoader,
        "pmlb": __load_pmlb_dataset,
        "imagenet": __load_imagenet_dataset,
        "synthetic": __load_synthetic_dataset,
    },
)
"""
    keras_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
    tensorflow_datasets = [
        'colorectal_histology', 'eurosat_all', 'eurosat_rgb',
        'horses_or_humans', 'patch_camelyon', 'places365_small'
    ]

    imagenet:
        imagenet_16
        imagenet_16_120
        imagenet_32
        imagenet_32_120
"""


def get_dataset_loader(source: str, name: str):
    return __source_loaders(source)(name)
