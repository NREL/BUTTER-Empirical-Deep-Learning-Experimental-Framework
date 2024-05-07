from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

import numpy
from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset_loader import DatasetLoader
from dmp.dataset.ml_task import MLTask


class TFImageClassificationDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, MLTask.classification)

    def _fetch_from_source(self) -> Dataset:
        import tensorflow_datasets
        import tensorflow

        datasets: List[tensorflow.data.Dataset] = tensorflow_datasets.load(
            self.dataset_name,
            split=["train", "test"],
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={
                "download_config": tensorflow_datasets.download.DownloadConfig(
                    verify_ssl=False
                )
            },
        )  # type: ignore

        results = []
        for split, ds in datasets.items():  # type: ignore
            ds.cache()
            inputs, outputs = [], []
            for input, output in tensorflow_datasets.as_numpy(ds):  # type: ignore
                inputs.append(input)
                outputs.append(output)
            results.append(DatasetGroup(numpy.array(inputs), numpy.array(outputs)))
        del datasets
        return Dataset(self.ml_task, *results)

    def _prepare_inputs(self, data):
        return self._prepare_image(data)
