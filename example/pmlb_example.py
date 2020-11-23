import math
import numbers
from pprint import pprint
from statistics import (
    mean,
    variance,
    )

import numpy
import pandas
import pmlb
import scipy
import tensorflow
from matplotlib import pyplot
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    )

from preprocessing.CategoricalIndexer import CategoricalIndexer
from preprocessing.Standardizer import Standardizer

'''
PMLB Homepage: https://github.com/EpistasisLab/pmlb
Python Reference: https://epistasislab.github.io/pmlb/python-ref.html



"537_houses" regression, 20640 observations, 8 features, 1 output
"529_pollen" regression, 3848 observations, 4 features, 1 output
"560_bodyfat" regression, 252 observations, 14 variables, 1 output


"adult" classification, 48842 observations, 14 features, 2 classes
"nursery" classification, 12958 observations, 8 features, 7 classes
"ring" classification, 7400 observations, 19 features, 2 classes
"satimage" classification, 6435 observations, 19 features, 6 classes
"cars"  classification, 392 observations, 8 features, 3 classes
"wine_recognition" classification, 178 observations, 13 features, 3 classes
"titanic" classification, 2201 observations, 3 features, 2 classes

4544_GeographicalOriginalofMusic	1059	117		continuous	0	regression

'''


def prepareValue(value):
    value = numpy.reshape(value, (-1, 1))
    pprint(value.shape)
    
    # TODO: Normalizer and PCA decorrelation can also help, etc
    # see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    # use one-hot when there are fewer distinct values than 10%
    # of the number of observations
    preprocessor = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor.fit(value)
    numDistinctValues = len(preprocessor.categories_[0])
    
    preparedValue = None
    if numDistinctValues <= 2:
        # if there are only two values, set them as 0 and 1
        preprocessed = (value == preprocessor.categories_[0][0])
    elif numDistinctValues / value.shape[0] > .1 and isinstance(value[0][0], numbers.Number):
        # use standardization otherwise
        # pprint(value)
        # m = numpy.mean(value)
        # s = numpy.std(value)
        # preprocessor = StandardScaler(with_mean=True, with_std=True)
        preprocessor = MinMaxScaler()
        preprocessor.fit(value)
        preprocessed = preprocessor.transform(value)
    else:
        preprocessed = preprocessor.transform(value)
    
    return preprocessed


def prepareMatrix(values):
    transformedList = []
    for i in range(values.shape[1]):
        value = values[:, i]
        transformedValue = prepareValue(value)
        transformedList.append(transformedValue)
    return numpy.hstack(transformedList)


# TODO: we are using dense numpy arrays, which could be very wasteful with sparse data


# inputs, outputs = pmlb.fetch_data('mushroom', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('537_houses', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('4544_GeographicalOriginalofMusic', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('505_tecator', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('201_pol', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('197_cpu_act', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('195_auto_price', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('542_pollution', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('503_wind', return_X_y=True)
# inputs, outputs = pmlb.fetch_data('527_analcatdata_election2000', return_X_y=True)
inputs, outputs = pmlb.fetch_data('560_bodyfat', return_X_y=True)
# how many iterations to reach crossover point -> normalize to number of nodes or number of weights?
# lowest validation error / validation error at crossover point
# vs
# depth, width
# could do grid search over both dims
# compare all vs best config?


# pprint(X)
# pprint(y)

pprint(inputs.shape)
pprint(outputs)
print(outputs.shape)

preparedInputs = prepareMatrix(inputs)
pprint(preparedInputs.shape)

preparedOutputs = prepareValue(outputs)
pprint(preparedOutputs.shape)

numObservations = preparedInputs.shape[0]
numInputs = preparedInputs.shape[1]
numOutputs = preparedOutputs.shape[1]

print('numObservations {} numInputs {} numOutputs {}'.format(numObservations, numInputs, numOutputs))

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Dense(16, activation=tensorflow.nn.relu, input_shape=(numInputs,)),  # input shape required
    tensorflow.keras.layers.Dense(16, activation=tensorflow.nn.relu),
    tensorflow.keras.layers.Dense(16, activation=tensorflow.nn.relu),
    tensorflow.keras.layers.Dense(numOutputs, activation=tensorflow.nn.sigmoid)
    ])

model.compile(
    # loss='binary_crossentropy', # binary classification
    # loss='categorical_crossentropy', # categorical classification (one hot)
    loss='mean_squared_error',  # regression
    optimizer=tensorflow.keras.optimizers.Adam(0.001),
    # optimizer='rmsprop',
    # metrics=['accuracy'],
    )

model.fit(
    x=preparedInputs,
    y=preparedOutputs,
    shuffle=True,
    validation_split=.2,
    # epochs=12,
    epochs=400,
    batch_size=256,
    )

losses = pandas.DataFrame(model.history.history)
losses.plot()
pyplot.show()
