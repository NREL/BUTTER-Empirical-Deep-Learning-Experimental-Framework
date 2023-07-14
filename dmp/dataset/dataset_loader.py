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

dataset_cache_directory = os.path.join(os.getcwd(), ".dataset_cache")


@dataclass
class DatasetLoader(ABC):
    """
    Responsible for returning a Dataset object when called.
    Contains convience functions for assisting in dataset loading and preprocessing.
    """

    source: str
    dataset_name: str
    ml_task: MLTask

    feature_prefix = "f"
    response_prefix = "r"
    index_delimiter = "x"
    _group_column = "g"

    def __call__(self) -> Dataset:
        """
        Generates a Dataset object (ususally by loading the dataset and preparing it for use)
        """
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
        filename = self.source + "_" + self.dataset_name + name
        return os.path.join(dataset_cache_directory, filename)

    def _try_read_from_cache(self) -> Optional[Dataset]:
        # filename = self._get_cache_path('.pkl.bz2')
        # try:
        #     import bz2
        #     with bz2.BZ2File(filename, 'rb') as file:
        #         return pickle.load(file)
        filename = self._get_cache_path(".pkl.lz4")
        try:
            import lz4

            with lz4.frame.open(filename, mode="rb") as file:
                return pickle.load(file)

        except FileNotFoundError:
            print(
                f"Dataset cache file {filename} not found while reading from dataset cache for {self}."
            )
            return None
        except:
            print(f"Error reading from dataset cache for {self}:")
            traceback.print_exc()
            try:
                os.remove(filename)
            except:
                print(f"Error removing bad cache file for {self}:")
                traceback.print_exc()
        return None

    @abstractmethod
    def _fetch_from_source(self) -> Dataset:
        pass

    def _write_to_cache(self, data: Dataset) -> None:
        try:
            # filename = self._get_cache_path('.pkl.bz2')
            # print(f'Writing dataset cache file {filename} for {self}.')

            # import bz2
            # with bz2.BZ2File(filename, 'wb', compresslevel=1) as file:
            #     pickle.dump(data, file)

            filename = self._get_cache_path(".pkl.lz4")
            print(f"Writing dataset cache file {filename} for {self}.")

            import lz4

            with lz4.frame.open(filename, mode="wb") as file:
                pickle.dump(data, file)

            print(f"Done writing dataset cache file {filename} for {self}.")

        except Exception as e:
            print(f"Error writing to dataset cache for {self}:")
            traceback.print_exc()

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
        raise Exception("Invalid shape {}.".format(shape))

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
        if num_distinct_values <= 20 or not isinstance(value[0][0], numbers.Number):
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
        value_transform: Callable[["DatasetLoader", ndarray], ndarray],
    ) -> Optional[ndarray]:
        value = numpy.reshape(value, (-1, 1))
        return value_transform(self, value)

    def prepare_matrix(
        self,
        values: ndarray,
        value_transform: Callable[["DatasetLoader", ndarray], ndarray],
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
        value_transform: Callable[["DatasetLoader", ndarray], ndarray],
    ) -> ndarray:
        # apply value_transform to all entries
        return value_transform(self, values)

    def min_max(self, value: ndarray) -> ndarray:
        preprocessor = MinMaxScaler()
        preprocessor.fit(value)
        return preprocessor.transform(value).astype(numpy.float32)

    def one_hot(self, value: ndarray) -> ndarray:
        preprocessor = OneHotEncoder(handle_unknown="ignore", sparse=False)
        preprocessor.fit(value)
        return preprocessor.transform(value).astype(numpy.int8)  # type: ignore

    def binary(self, value: ndarray) -> ndarray:
        # if there are only two values, set them as 0 and 1
        return (value == value[0]).astype(numpy.int8)  # type: ignore

    def _prepare_image(self, data: ndarray) -> ndarray:
        return (
            data.astype(numpy.float16) / numpy.array(255.0, dtype=numpy.float16)
        ).astype(numpy.float16)


def load_dataset_index(path: str) -> pandas.DataFrame:
    datasets = pandas.read_csv(path)
    datasets.set_index("Dataset", inplace=True, drop=False)
    return datasets
