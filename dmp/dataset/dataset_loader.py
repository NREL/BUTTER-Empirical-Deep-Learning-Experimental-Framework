from itertools import chain
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
import json
import os
import pickle
import traceback
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Any,
)
import numpy
from numpy import ndarray
import pandas
import pyarrow
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset import Dataset
from dmp.dataset.ml_task import MLTask
from dmp.parquet_util import make_pyarrow_table_from_numpy

dataset_cache_directory = os.path.join(os.getcwd(), '.dataset_cache')


@dataclass
class DatasetLoader(ABC):

    dataset_name: str
    ml_task: MLTask

    feature_prefix = 'f'
    response_prefix = 'r'
    index_delimiter = 'x'
    _group_column = 'g'

    def __call__(self) -> Dataset:
        # check cache first for raw inputs and outputs in the working directory
        # cache_directory = os.path.join(os.getcwd(), '.dataset_cache', self.dataset_name)
        # os.makedirs(self.dataset_cache_directory, exist_ok=True)
        # raw_inputs, raw_outputs = self.read_from_source(cache_directory, self.dataset_name)
        """ See if the file has been cached and try to read that, download otherwise"""
        data = self._load_dataset()
        data = self._prepare_dataset_data(data)
        return data

    def _load_dataset(self):
        data = self._try_read_from_cache()
        if data is None:
            data = self._fetch_from_source()
            self._write_to_cache(data)
        return data

    def _get_cache_path(self, name):
        filename = self.dataset_name + f'_{name}'
        return os.path.join(dataset_cache_directory, filename)

    def _try_read_from_cache(self) -> Optional[Dataset]:
        filename = self._get_cache_path('.pq')
        try:
            os.makedirs(dataset_cache_directory, exist_ok=True)
            table = pyarrow.parquet.read_from_file(filename)

            def accumulate_variable_name(prefix, name):
                index_string = name[len(prefix):]
                return tuple(
                    (int(i) for i in index_string.split(self.index_delimiter)))

            def accumulate_columns(prefix):
                columns = [(name, accumulate_variable_name(prefix, name))
                           for name in table.column_names()
                           if name.starts_with(prefix)]
                columns.sort(key=lambda c: c[1])
                shape = columns[-1][1]
                return columns, shape

            def extract_columns(prefix, group_code) -> numpy.ndarray:
                group_table = pyarrow.compute.equal(
                    table[self._group_column],
                    group_code,
                )
                columns, shape = accumulate_columns(prefix)
                result = None
                if len(columns) == 1:
                    result = group_table[columns[0][0]].to_numpy()
                else:
                    result = numpy.hstack([
                        group_table[column].to_numpy()
                        for column, index in columns
                    ])
                return result.reshape((group_table.shape[0], *shape))

            return Dataset(
                self.ml_task,
                *(DatasetGroup(
                    extract_columns(self.feature_prefix, group_code),
                    extract_columns(self.response_prefix, group_code),
                ) for group_code in range(3)),
            )

        except FileNotFoundError:
            return None
        except:
            print(f'Error reading from dataset cache for {self}:')
            traceback.print_exc()
            try:
                os.remove(filename)
            except:
                print(f'Error removing bad cache file for {self}:')
                traceback.print_exc()
        return None

    @abstractmethod
    def _fetch_from_source(self) -> Dataset:
        pass

    def _write_to_cache(self, data: Dataset) -> None:
        # with open(self._get_cache_path('.pkl'), 'wb') as file_handle:
        #     pickle.dump(data, file_handle)
        splits = [
            (group_code, dataset_group)
            for group_code, (prefix,
                             dataset_group) in enumerate(data.full_splits)
            if dataset_group is not None
        ]

        group_column = numpy.vstack(
            list(
                numpy.repeat(group_code, dataset_group.inputs.shape[0]).astype(
                    numpy.int8) for group_code, dataset_group in splits))

        variable_types = [
                             (self.feature_prefix,
                              lambda dataset_group: dataset_group.inputs),
                             (self.response_prefix,
                              lambda dataset_group: dataset_group.outputs),
                         ]

        all_columns = []
        all_arrays = []

        for prefix, getter in variable_types:
            shape = None
            indexes = []
            arrays = []
            columns = []
            for group_code, dataset_group in splits:
                array = getter(dataset_group)
                if len(array.shape) == 1:
                    array = array.reshape((array.shape[0], 1))

                if shape is None:
                    shape = array.shape[1:]
                    print(f'shape: {array.shape} : {shape}')
                    indexes = list(numpy.ndindex(shape))
                    arrays = [ [], ] * len(indexes)
                    columns = [prefix + self.index_delimiter.join((str(i) for i in index)) for index in indexes]
                
                for i, index in enumerate(indexes):
                    arrays[i].append(array[(slice(None), *index)])
            
            all_columns.extend(columns)
            all_arrays.extend(arrays)

        numpy_columns = [numpy.vstack(arrays) for arrays in all_arrays]
        print(numpy_columns)
        print(all_arrays)

        table, use_byte_stream_split = make_pyarrow_table_from_numpy(
            all_columns, 
            numpy_columns,
        )
        del all_arrays

        pyarrow.parquet.write_table(table, self._get_cache_path('.pq'),
                compression='ZSTD',
                compression_level=15,
                use_dictionary=False,
                use_byte_stream_split=use_byte_stream_split,  # type: ignore
                version='2.6',
                data_page_version='2.0',
                write_statistics=False,
        )

        

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
        return self.prepare_data(data, self.dynamic_value_transform)

    def _prepare_outputs(self, data) -> ndarray:
        return self.prepare_data(data, self.dynamic_output_value_transform)

    def prepare_data(self, value, transform) -> ndarray:
        shape = value.shape
        if len(shape) == 1:
            return self.prepare_value(value, transform)  # type: ignore
        elif len(shape) > 1:
            return self.prepare_matrix(value, transform)  # type: ignore
        raise Exception('Invalid shape {}.'.format(shape))

    @staticmethod
    def dynamic_value_transform(
        self,
        value: ndarray,
    ) -> Optional[ndarray]:
        # TODO: Normalizer and PCA decorrelation can also help, etc
        # see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        # use one-hot when there are 20 or fewer distinct values, or the
        # values are not numbers

        num_distinct_values = numpy.unique(value).size
        if num_distinct_values <= 1:
            return None  # ignore it
        if num_distinct_values <= 2:
            return self.binary(value)
        if num_distinct_values <= 20 or \
            not isinstance(value[0][0], numbers.Number):
            return self.one_hot(value)
        return self.min_max(value)

    @staticmethod
    def dynamic_output_value_transform(
        self,
        value: ndarray,
    ) -> Optional[ndarray]:
        if self.ml_task == MLTask.classification:
            num_distinct_values = numpy.unique(value).size
            if num_distinct_values <= 1:
                return numpy.zeros_like(value)
            if num_distinct_values <= 2:
                return self.binary(value)
            return self.one_hot(value)
        else:
            return self.min_max(value)

    def prepare_value(
        self,
        value: ndarray,
        value_transform: Callable[['DatasetLoader', ndarray], ndarray],
    ) -> Optional[ndarray]:
        value = numpy.reshape(value, (-1, 1))
        return value_transform(self, value)

    def prepare_matrix(
        self,
        values: ndarray,
        value_transform: Callable[['DatasetLoader', ndarray], ndarray],
    ) -> Optional[ndarray]:
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
        value_transform: Callable[['DatasetLoader', ndarray], ndarray],
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
        return (value == value[0]).astype(numpy.int)  # type: ignore

    def _prepare_image(self, data: ndarray) -> ndarray:
        return (data / 255.0).astype(numpy.float16)


def load_dataset_index(path: str) -> pandas.DataFrame:
    datasets = pandas.read_csv(path)
    datasets.set_index('Dataset', inplace=True, drop=False)
    return datasets
