from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

import numpy
import numpy as np
from numpy import ndarray
import pandas
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)
from dmp.dataset.ml_task import MLTask

dataset_cache_directory = os.path.join(os.getcwd(), '.dataset_cache')

@dataclass
class DatasetLoader(ABC):

    dataset_name: str
    ml_task: MLTask

    def __call__(self):
        # check cache first for raw inputs and outputs in the working directory
        # cache_directory = os.path.join(os.getcwd(), '.dataset_cache', self.dataset_name)
        os.makedirs(dataset_cache_directory, exist_ok=True)
        # raw_inputs, raw_outputs = self.read_from_source(cache_directory, self.dataset_name)
        """ See if the file has been cached and try to read that, download otherwise"""
        filename = self.dataset_name + '.npy'
        file_path = os.path.join(dataset_cache_directory, filename)
        data = self._try_read_from_cache()
        if data is None:
            data = self._prepare(self._fetch_from_source())
            self._write_to_cache(data)
        return data

    def _get_cache_path(self):
        return os.path.join(dataset_cache_directory, self.dataset_name + '.npy')

    def _try_read_from_cache(self):
        os.makedirs(dataset_cache_directory, exist_ok=True)
        raw_inputs, raw_outputs = None, None
        try:
            with open(self._get_cache_path(), 'rb') as f:
                raw_inputs = numpy.load(f, allow_pickle=True)
                raw_outputs = numpy.load(f, allow_pickle=True)
        except FileNotFoundError:
            return None
        return (raw_inputs, raw_outputs)

    @abstractmethod
    def _fetch_from_source(self):
        pass

    def _write_to_cache(self, data):
        with open(self._get_cache_path(), 'wb') as f:
            for d in data:
                numpy.save(f, d)

    def _prepare(self, data) -> Tuple[ndarray, ndarray]:
        return tuple((self.prepare_data(d) for d in data))

    def prepare_data(self, value) -> ndarray:
        shape = value.shape
        if len(shape) == 1:
            return self.prepare_value(value, )
        elif len(shape) > 1:
            return self.prepare_matrix(value)
        raise Exception('Invalid shape {}.'.format(shape))

    def dynamic_value_transform(self, value: ndarray) -> ndarray:
        # TODO: Normalizer and PCA decorrelation can also help, etc
        # see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        # use one-hot when there are fewer distinct values than 10%
        # of the number of observations
        if self.ml_task == MLTask.classification:
            num_distinct_values = numpy.unique(value).size
            preprocessed = None
            if num_distinct_values <= 1:
                raise NotImplementedError()
            if num_distinct_values <= 2:
                return self.binary(value)
            return self.one_hot(value)
        else:
            preprocessed = self.min_max(value)
        return preprocessed

    def prepare_value(
        self,
        value: ndarray,
        value_transform: Callable[['DatasetLoader', ndarray],
                                  ndarray] = dynamic_value_transform,
    ) -> ndarray:
        value = numpy.reshape(value, (-1, 1))
        return value_transform(self, value)

    def prepare_matrix(
        self,
        values: ndarray,
        value_transform: Callable[['DatasetLoader', ndarray],
                                  ndarray] = dynamic_value_transform,
    ) -> ndarray:
        transformed_list = []
        for i in range(values.shape[1]):
            value = values[:, i]
            transformed_value = self.prepare_value(value, value_transform)
            if transformed_value is not None:
                transformed_list.append(transformed_value)
        return numpy.hstack(transformed_list)

    def prepare_tensor(
        self,
        values: ndarray,
        value_transform: Callable[['DatasetLoader', ndarray],
                                  ndarray] = dynamic_value_transform,
    ) -> ndarray:
        # apply value_transform to all entries
        return value_transform(self, values)

    def min_max(self, value: ndarray) -> ndarray:
        preprocessor = MinMaxScaler()
        preprocessor.fit(value)
        return preprocessor.transform(value)

    def one_hot(self, value: ndarray) -> ndarray:
        preprocessor = OneHotEncoder(handle_unknown='ignore', sparse=False)
        preprocessor.fit(value)
        return preprocessor.transform(value)  # type: ignore

    def binary(self, value: ndarray) -> ndarray:
        # if there are only two values, set them as 0 and 1
        return (value == value[0]).astype(np.int)  # type: ignore

    def _prepare_image(self, data) -> Tuple[ndarray, ndarray]:
        return data[0] / 255.0, self.prepare_value(data[1])


def load_dataset_index(path: str) -> pandas.DataFrame:
    datasets = pandas.read_csv(path)
    datasets.set_index('Dataset', inplace=True, drop=False)
    return datasets
