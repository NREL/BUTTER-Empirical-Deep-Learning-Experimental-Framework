import numbers
from typing import Optional

import numpy
import pandas
import pmlb
from numpy import ndarray
from pandas import Series
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    )


def loadDatasetIndex(filePath: str = 'data/pmlb/pmlb.csv') -> pandas.DataFrame:
    datasets = pandas.read_csv(filePath)
    datasets.set_index('Dataset', inplace=True, drop=False)
    return datasets


def loadData(dataset: Series) -> (ndarray, ndarray):
    rawInputs, rawOutputs = pmlb.fetch_data(dataset['Dataset'], return_X_y=True)
    
    inputs = _prepareData(rawInputs)
    outputs = _prepareData(rawOutputs)
    
    return inputs, outputs


def _prepareData(value) -> ndarray:
    shape = value.shape
    if len(shape) == 1:
        return _prepareValue(value)
    if len(shape) > 1:
        return _prepareMatrix(value)
    raise Exception('Invalid shape {}.'.format(shape))


def _prepareValue(value) -> Optional[ndarray]:
    value = numpy.reshape(value, (-1, 1))
    # pprint(value.shape)
    
    # TODO: Normalizer and PCA decorrelation can also help, etc
    # see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    # use one-hot when there are fewer distinct values than 10%
    # of the number of observations
    
    numDistinctValues = numpy.unique(value).size
    preprocessor = None
    preparedValue = None
    if numDistinctValues <= 1:
        return None
    elif numDistinctValues <= 2:
        # if there are only two values, set them as 0 and 1
        preprocessed = (value == value[0])
    elif numDistinctValues / value.shape[0] < .1 and isinstance(value[0][0], numbers.Number):
        # use standardization otherwise
        # pprint(value)
        # m = numpy.mean(value)
        # s = numpy.std(value)
        # preprocessor = StandardScaler(with_mean=True, with_std=True)
        preprocessor = MinMaxScaler()
        preprocessor.fit(value)
        preprocessed = preprocessor.transform(value)
    else:
        preprocessor = OneHotEncoder(handle_unknown='ignore', sparse=False)
        preprocessor.fit(value)
        preprocessed = preprocessor.transform(value)
    
    return preprocessed


def _prepareMatrix(values: ndarray) -> ndarray:
    transformedList = []
    for i in range(values.shape[1]):
        value = values[:, i]
        transformedValue = _prepareValue(value)
        if transformedValue is not None:
            transformedList.append(transformedValue)
    return numpy.hstack(transformedList)
