from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
import json
import os
import traceback
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
import pyarrow
import pyarrow.compute as pc
import numpy
import numpy as np
from numpy import ndarray
import pandas
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset import Dataset
from dmp.dataset.ml_task import MLTask
from dmp.parquet_util import make_pyarrow_schema

dataset_cache_directory = os.path.join(os.getcwd(), '.dataset_cache')


@dataclass
class DatasetLoader(ABC):

    dataset_name: str
    ml_task: MLTask

    def __call__(self) -> Dataset:
        # check cache first for raw inputs and outputs in the working directory
        # cache_directory = os.path.join(os.getcwd(), '.dataset_cache', self.dataset_name)
        os.makedirs(dataset_cache_directory, exist_ok=True)
        # raw_inputs, raw_outputs = self.read_from_source(cache_directory, self.dataset_name)
        """ See if the file has been cached and try to read that, download otherwise"""
        data = self._try_read_from_cache()
        if data is None:
            data = self._prepare_dataset_data(self._fetch_from_source())
            self._write_to_cache(data)
            self.ml_task = data.ml_task
        return data

    def _get_cache_path(self, name):
        filename = self.dataset_name + f'_{name}'
        return os.path.join(dataset_cache_directory, filename)

    def _try_read_from_cache(self) -> Optional[Dataset]:
        lockfile_name = self._get_cache_path('lock')
        try:
            os.makedirs(dataset_cache_directory, exist_ok=True)
            array_dict = numpy.load(self._get_cache_path('.npz'))

            def get_group(name):
                input_name = f'{name}_inputs'
                output_name = f'{name}_outputs'
                if input_name in array_dict:
                    return DatasetGroup(
                        array_dict[input_name],
                        array_dict[output_name],
                    )
                return None

            return Dataset(
                self.ml_task,
                get_group('train'),
                get_group('test'),
                get_group('validation'),
            )
        except FileNotFoundError:
            return None
        except:
            print(f'Error reading from dataset cache for {self}:')
            traceback.print_exc()
            try:
                os.remove(lockfile_name)
            except:
                print(f'Error removing bad cache file for {self}:')
                traceback.print_exc()
        return None

    @abstractmethod
    def _fetch_from_source(self) -> Dataset:
        pass

    def _write_to_cache(self, data: Dataset) -> None:
        array_dict = {}
        for name, split in data.splits:
            array_dict[f'{name}_inputs'] = split.inputs
            array_dict[f'{name}_outputs'] = split.outputs

        np.savez(self._get_cache_path('.npz'), **array_dict)

    def _prepare_dataset_data(self, data: Dataset) -> Dataset:
        data.train = self._prepare_data_group(data.train)
        data.test = self._prepare_data_group(data.test)
        data.validation = self._prepare_data_group(data.validation)
        return data

    def _prepare_data_group(
        self,
        group: Optional[DatasetGroup],
    ) -> Optional[DatasetGroup]:
        if group is None:
            return None
        group.inputs = self._prepare_inputs(group.inputs)
        group.outputs = self._prepare_outputs(group.outputs)
        return group

    def _prepare_inputs(self, data) -> ndarray:
        return self.prepare_data(data)

    def _prepare_outputs(self, data) -> ndarray:
        return self.prepare_data(data)

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

    def _prepare_image(self, data: ndarray) -> ndarray:
        return (data / 255.0).astype(numpy.float16)


def load_dataset_index(path: str) -> pandas.DataFrame:
    datasets = pandas.read_csv(path)
    datasets.set_index('Dataset', inplace=True, drop=False)
    return datasets
