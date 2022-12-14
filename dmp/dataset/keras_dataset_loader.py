from re import I
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


class KerasDatasetLoader(DatasetLoader):

    def __init__(
        self,
        dataset_name: str,
        keras_load_data_function: Callable,
    ) -> None:
        super().__init__(dataset_name, MLTask.classification)
        self._keras_load_data_function: Callable = keras_load_data_function

    def _fetch_from_source(self):
        (xtrain, ytrain), (xtest, ytest) = self._keras_load_data_function()
        raw_inputs = numpy.concatenate((xtrain, xtest), axis=0)
        raw_outputs = numpy.concatenate((ytrain, ytest), axis=0)
        return raw_inputs, raw_outputs

    def _prepare(self, data):
        return self._prepare_image(data)