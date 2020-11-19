import math
from pprint import pprint
from statistics import (
    mean,
    variance,
    )

import numpy
import pmlb
import scipy
import tensorflow
from sklearn.preprocessing import (
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

'''


def prepareValue(value):
    value = numpy.reshape(value, (-1, 1))
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
    elif numDistinctValues / value.shape[0] > .1:
        # use standardization otherwise
        m = mean(value)
        s = math.sqrt(variance(value))
        preprocessor = StandardScaler(with_mean=m, with_std=s)
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


inputs, outputs = pmlb.fetch_data('mushroom', return_X_y=True)

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
    tensorflow.keras.layers.Dense(20, activation=tensorflow.nn.relu, input_shape=(numInputs,)),  # input shape required
    tensorflow.keras.layers.Dense(20, activation=tensorflow.nn.relu),
    tensorflow.keras.layers.Dense(numOutputs, activation=tensorflow.nn.sigmoid)
    ])

model.compile(
    loss='binary_crossentropy',
    optimizer=tensorflow.keras.optimizers.Adam(0.001),
    # optimizer='rmsprop',
    metrics=['accuracy'],
)

model.fit(
    x=preparedInputs,
    y=preparedOutputs,
    shuffle=True,
    validation_split=.2,
    # epochs=12,
    epochs = 50,
    batch_size = 256,
)

