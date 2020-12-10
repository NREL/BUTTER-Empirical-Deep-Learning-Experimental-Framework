"""

"""
import gc
import json
import math
import os
import sys
from copy import deepcopy

import numpy
import pandas
import tensorflow
from tensorflow.keras import (
    callbacks,
    losses,
    Sequential,
    metrics,
    optimizers,
    )
from tensorflow.python.keras.models import Model

from command_line_tools import (
    command_line_config,
    run_tools,
    )
from dmp.data.pmlb import PMLBLoader
from dmp.data.pmlb.PMLBLoader import loadDataset


def countTrainableParameters(model: Model) -> int:
    count = 0
    for var in model.trainable_variables:
        print('ctp {}'.format(var.get_shape()))
        acc = 1
        for dim in var.get_shape():
            acc *= int(dim)
        print('ctp acc {}'.format(acc))
        count += acc
    print('ctp total {}'.format(count))
    return count


class NpEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def testNetwork(
        config: {},
        dataset,
        inputs: numpy.ndarray,
        outputs: numpy.ndarray,
        prefix,
        width: int,
        depth: int,
        ) -> None:
    config = deepcopy(config)
    name = '{}_{}_{}_b_{}_w_{}_d_{}'.format(
        dataset['Task'],
        dataset['Endpoint'],
        dataset['Dataset'],
        prefix,
        width,
        depth)
    
    config['name'] = name
    config['depth'] = depth
    config['numHidden'] = max(0, depth - 2)
    config['width'] = width
    
    # pprint(config)
    runName = run_tools.get_run_name(config)
    config['runName'] = runName
    
    numObservations = inputs.shape[0]
    numInputs = inputs.shape[1]
    numOutputs = outputs.shape[1]
    
    logData = {'config': config}
    runConfig = config['runConfig']
    
    runOptimizer = optimizers.Adam(0.001)
    runMetrics = [
        # metrics.CategoricalAccuracy(),
        'accuracy',
        metrics.CosineSimilarity(),
        metrics.Hinge(),
        metrics.KLDivergence(),
        metrics.MeanAbsoluteError(),
        metrics.MeanSquaredError(),
        metrics.MeanSquaredLogarithmicError(),
        metrics.RootMeanSquaredError(),
        metrics.SquaredHinge(),
        ]
    
    print('input shape {} output shape {}'.format(inputs.shape, outputs.shape))
    # print(inputs[0, :])
    # print(outputs[0, :])
    runLoss = losses.mean_squared_error
    outputActivation = tensorflow.nn.relu
    runTask = dataset['Task']
    if runTask == 'regression':
        runLoss = losses.mean_squared_error
        outputActivation = tensorflow.nn.sigmoid
        print('mean_squared_error')
    elif runTask == 'classification':
        outputActivation = tensorflow.nn.softmax
        if numOutputs == 1:
            runLoss = losses.binary_crossentropy
            print('binary_crossentropy')
        else:
            runLoss = losses.categorical_crossentropy
            print('categorical_crossentropy')
    else:
        raise Exception('Unknown task "{}"'.format(runTask))
    
    layers = []
    for d in range(depth):
        
        if d == depth - 1:
            # output layer
            layerWidth = numOutputs
            activation = outputActivation
        else:
            layerWidth = width
            activation = tensorflow.nn.relu
        
        layer = None
        if d == 0:
            # input layer
            layer = tensorflow.keras.layers.Dense(
                layerWidth,
                activation=activation,
                input_shape=(numInputs,))
        else:
            layer = tensorflow.keras.layers.Dense(
                layerWidth,
                activation=activation,
                )
        print('d {} w {} in {}'.format(d, layerWidth, numInputs))
        layers.append(layer)
    
    model = Sequential(layers)
    model.compile(
        # loss='binary_crossentropy', # binary classification
        # loss='categorical_crossentropy', # categorical classification (one hot)
        loss=runLoss,  # regression
        optimizer=runOptimizer,
        # optimizer='rmsprop',
        # metrics=['accuracy'],
        metrics=runMetrics,
        )
    
    logData['numWeights'] = countTrainableParameters(model)
    logData['numInputs'] = numInputs
    logData['numFeatures'] = dataset['n_features']
    logData['numClasses'] = dataset['n_classes']
    logData['numOutputs'] = numOutputs
    logData['numObservations'] = numObservations
    logData['task'] = dataset['Task']
    logData['endpoint'] = dataset['Endpoint']
    
    runCallbacks = [
        callbacks.EarlyStopping(**config['earlyStopping']),
        ]
    
    gc.collect()
    
    historyCallback = model.fit(
        x=inputs,
        y=outputs,
        callbacks=runCallbacks,
        **runConfig,
        )
    
    history = historyCallback.history
    logData['history'] = history
    
    validationLosses = numpy.array(history['val_loss'])
    bestIndex = numpy.argmin(validationLosses)
    
    logData['iterations'] = bestIndex + 1
    logData['val_loss'] = validationLosses[bestIndex]
    logData['loss'] = history['loss'][bestIndex]
    
    logPath = config['logPath']
    run_tools.makedir_if_not_exists(logPath)
    logFile = os.path.join(logPath, '{}.json'.format(runName))
    print('log file: {}'.format(logFile))
    
    with open(logFile, 'w', encoding='utf-8') as f:
        json.dump(logData, f, ensure_ascii=False, indent=2, sort_keys=True, cls=NpEncoder)


def testAspectRatio(config, dataset, inputs, outputs, budget, depths):
    config = deepcopy(config)
    config['dataset'] = dataset['Dataset'],
    config['datasetRow'] = list(dataset),
    
    config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
    # inputs, outputs = PMLBLoader.loadData(dataset)
    gc.collect()
    
    # for width in range(1, 128):
    #     for depth in range(1, 32):
    for depth in depths:
        i = inputs.shape[1]
        h = (depth - 2)
        o = outputs.shape[1]
        
        a = h
        b = i + h + o + 1
        c = o - budget
        
        rawWidth = 1
        if h == 0:
            rawWidth = -(o - budget) / (i + o + 1)
        else:
            rawWidth = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        width = round(rawWidth)
        print('budget {} depth {}, i {} h {} o {}, a {} b {} c {}, rawWidth {}, width {}'.format(budget, depth, i, h, o,
                                                                                                 a, b, c, rawWidth,
                                                                                                 width))
        testNetwork(config, dataset, inputs, outputs, '{}'.format(budget), width, depth)
    
    # pprint(logData)
    print('done.')
    gc.collect()


pandas.set_option("display.max_rows", None, "display.max_columns", None)
datasets = PMLBLoader.loadDatasetIndex()

# core_config = tensorflow.Conf()
# core_config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=core_config)
# tensorflow.keras.backend.set_session(session)


# for index, dataset in datasets.iterrows():
#     print(index)

default_config = {
    'logPath':       '/home/ctripp/log',
    'earlyStopping': {
        'patience':             10,
        'monitor':              'val_loss',
        'min_delta':            0,
        'verbose':              0,
        'mode':                 'min',
        'baseline':             None,
        'restore_best_weights': False,
        },
    'runConfig':     {
        'validation_split': .2,
        'shuffle':          True,
        'epochs':           10000,
        'batch_size':       256,
        },
    'activation':    'relu',
    }

config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)

# dataset, inputs, outputs = loadDataset(datasets, 'mnist')
# dataset, inputs, outputs = loadDataset(datasets, '537_houses')
# for i in [.125, .25, .5, 1, 2, 4, 8, 16, 32]:
#     budget = int(round(i * 1000))
#     for _ in range(50):
#         testAspectRatio(config, dataset, inputs, outputs, budget, [i for i in range(2, 20)])
  
dataset, inputs, outputs = loadDataset(datasets, 'mnist')
for _ in range(20):
    for i in [16, 32, 64, 128, 256]:
        budget = int(round(i * 1000))
        testAspectRatio(config, dataset, inputs, outputs, budget, [i for i in range(2, 20)])
# testAspectRatio(config, dataset, inputs, outputs, 128, [i for i in range(2, 16)])

# pprint(logData)
print('done.')
gc.collect()
