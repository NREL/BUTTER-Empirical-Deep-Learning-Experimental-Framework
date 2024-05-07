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


class KerasImageDatasetLoader(DatasetLoader):
    def __init__(
        self,
        dataset_name: str,
        keras_load_data_function: Callable,
    ) -> None:
        super().__init__("keras", dataset_name, MLTask.classification)
        self._keras_load_data_function: Callable = keras_load_data_function

    def _fetch_from_source(self):
        train, test = self._keras_load_data_function()
        return Dataset(
            self.ml_task,
            DatasetGroup(*train),
            DatasetGroup(*test),
        )

    def _prepare_inputs(self, data):
        return self._prepare_image(data)
