import numbers
import os
from pprint import pprint
from typing import (
    Callable,
    Optional,
    Tuple,
    )

import numpy
import pandas
import pmlb
from numpy import ndarray
from pandas import Series
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    )


dataset_path = os.path.join(os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))), 'pmlb.csv')

def loadDatasetIndex(filePath: str = dataset_path) -> pandas.DataFrame:
    datasets = pandas.read_csv(filePath)
    datasets.set_index('Dataset', inplace=True, drop=False)
    return datasets


def loadDataset(datasets: pandas.DataFrame, datasetName: str) -> (pandas.Series, ndarray, ndarray):
    matchingDatasets = datasets[datasets['Dataset'] == datasetName]
    if len(matchingDatasets) <= 0:
        raise Exception('No matching dataset "{}".'.format(datasetName))
    dataset = matchingDatasets.iloc[0]
    rawInputs, rawOutputs = pmlb.fetch_data(datasetName, return_X_y=True)
    
    loader = _defaultLoader
    if datasetName in _customLoaders:
        loader = _customLoaders[datasetName]
    
    return dataset, *loader(rawInputs, rawOutputs)


def _defaultLoader(rawInputs: ndarray, rawOutputs: ndarray) -> (ndarray, ndarray):
    inputs = _prepareData(rawInputs)
    outputs = _prepareData(rawOutputs)
    return inputs, outputs


def _loadMNIST(rawInputs: ndarray, rawOutputs: ndarray) -> (ndarray, ndarray):
    inputs = _prepareMatrix(rawInputs, lambda value: value / 255.0)
    outputs = _prepareValue(rawOutputs, _oneHot)
    pprint(outputs)
    return inputs, outputs


_customLoaders: {str: Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]]} = {
    'mnist': _loadMNIST,
    }


def _prepareData(value) -> ndarray:
    shape = value.shape
    if len(shape) == 1:
        return _prepareValue(value)
    if len(shape) > 1:
        return _prepareMatrix(value)
    raise Exception('Invalid shape {}.'.format(shape))


def _dynamicValueTransform(value: ndarray) -> Optional[ndarray]:
    # TODO: Normalizer and PCA decorrelation can also help, etc
    # see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    # use one-hot when there are fewer distinct values than 10%
    # of the number of observations
    
    numDistinctValues = numpy.unique(value).size
    preprocessed = None
    if numDistinctValues <= 1:
        return None
    elif numDistinctValues <= 2:
        preprocessed = _binary(value)
    elif numDistinctValues <= 20 or not isinstance(value[0][0], numbers.Number):  # numDistinctValues > .01 * value.shape[0]
        preprocessed = _oneHot(value)
    else:
        preprocessed = _minMax(value)
    
    return preprocessed


def _prepareValue(
        value: ndarray,
        valueTransform: Optional[Callable[[ndarray], ndarray]] = _dynamicValueTransform,
        ) -> Optional[ndarray]:
    value = numpy.reshape(value, (-1, 1))
    return valueTransform(value)


def _prepareMatrix(
        values: ndarray,
        valueTransform: Optional[Callable[[ndarray], ndarray]] = _dynamicValueTransform,
        ) -> ndarray:
    transformedList = []
    for i in range(values.shape[1]):
        value = values[:, i]
        transformedValue = _prepareValue(value, valueTransform)
        if transformedValue is not None:
            transformedList.append(transformedValue)
    return numpy.hstack(transformedList)


def _minMax(value: ndarray) -> ndarray:
    preprocessor = MinMaxScaler()
    preprocessor.fit(value)
    return preprocessor.transform(value)


def _oneHot(value: ndarray) -> ndarray:
    preprocessor = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor.fit(value)
    return preprocessor.transform(value)


def _binary(value: ndarray) -> ndarray:
    # if there are only two values, set them as 0 and 1
    return value == value[0]
