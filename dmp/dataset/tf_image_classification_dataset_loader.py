from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

import numpy
from dmp.dataset.dataset_loader import DatasetLoader
from dmp.dataset.ml_task import MLTask



class TFImageClassificationDatasetLoader(DatasetLoader):

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, MLTask.classification)

    def _fetch_from_source(self):
        import tensorflow_datasets
        dl_config = tensorflow_datasets.download.DownloadConfig(
            verify_ssl=False)

        datasets = tensorflow_datasets.load(
            self.dataset_name,
            split=None,
            shuffle_files=False,
            as_supervised=True,
            download_and_prepare_kwargs={'download_config': dl_config})

        inputs, outputs = [], []
        for ds in datasets.values():  # type: ignore
            for input, output in tensorflow_datasets.as_numpy(
                    ds):  # type: ignore
                inputs.append(input)
                outputs.append(output)

        return numpy.array(inputs), numpy.array(outputs)

    def _prepare(self, data):
        return self._prepare_image(data)
