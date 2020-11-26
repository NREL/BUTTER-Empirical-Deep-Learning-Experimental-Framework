"""

"""
import gc
import json
import os
import sys
from pprint import pprint

import numpy
import pandas
import tensorflow
import tensorflow_datasets
from matplotlib import pyplot
from pathos import multiprocessing
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
from command_line_tools.run_tools import setup_run
from data.pmlb import PMLBLoader


def countTrainableParameters(model: Model) -> int:
    count = 0
    for var in model.trainable_variables:
        acc = 1
        for dim in var.get_shape():
            acc *= int(dim)
        count += acc
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


pandas.set_option("display.max_rows", None, "display.max_columns", None)
datasets = PMLBLoader.loadDatasetIndex()

# core_config = tensorflow.Conf()
# core_config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=core_config)
# tensorflow.keras.backend.set_session(session)


# datasets = datasets[datasets['Dataset'] == 'mnist']
for index, dataset in datasets.iterrows():
    print(index)
    
    default_config = {
        'logPath':       '/home/ctripp/log',
        'dataset':       dataset['Dataset'],
        'datasetRow':    list(dataset),
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
    inputs, outputs = PMLBLoader.loadData(dataset)
    gc.collect()
    numObservations = inputs.shape[0]
    numInputs = inputs.shape[1]
    numOutputs = outputs.shape[1]
    
    for width in range(1, 128):
        for depth in range(1, 32):
            
            name = '{}_{}_{}_w_{}_d_{}'.format(
                dataset['Task'],
                dataset['Endpoint'],
                dataset['Dataset'],
                width,
                depth)
            
            config['name'] = name
            config['depth'] = depth
            config['width'] = width
            
            # pprint(config)
            runName = run_tools.get_run_name(config)
            config['runName'] = runName
            depth = config['depth']
            width = config['width']
            
            logData = {'config': config}
            runConfig = config['runConfig']
            
            runLoss = losses.mean_squared_error
            runOptimizer = optimizers.Adam()
            runMetrics = [
                metrics.Accuracy(),
                metrics.CosineSimilarity(),
                metrics.Hinge(),
                metrics.KLDivergence(),
                metrics.MeanAbsoluteError(),
                metrics.MeanSquaredError(),
                metrics.MeanSquaredLogarithmicError(),
                metrics.RootMeanSquaredError(),
                metrics.SquaredHinge(),
                ]
            
            runTask = dataset['Task']
            if runTask == 'regression':
                runLoss = losses.mean_squared_error
            elif runTask == 'classification':
                if numOutputs == 1:
                    runLoss = losses.binary_crossentropy
                else:
                    runLoss = losses.categorical_crossentropy
            else:
                raise Exception('Unknown task "{}"'.format(runTask))
            
            layers = []
            for d in range(depth):
                
                if d == depth - 1:
                    # output layer
                    layerWidth = numOutputs
                    activation = tensorflow.nn.sigmoid
                else:
                    layerWidth = width
                    activation = tensorflow.nn.relu
                
                layer = None
                if d == 0:
                    layer = tensorflow.keras.layers.Dense(
                        layerWidth,
                        activation=activation,
                        input_shape=(numInputs,))
                else:
                    layer = tensorflow.keras.layers.Dense(
                        layerWidth,
                        activation=activation,
                        )
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
            logData['numWeights'] = countTrainableParameters(model)
            logData['numInputs'] = numInputs
            logData['numFeatures'] = dataset['n_features']
            logData['numClasses'] = dataset['n_classes']
            logData['numOutputs'] = numOutputs
            logData['numObservations'] = numObservations
            logData['task'] = dataset['Task']
            logData['endpoint'] = dataset['Endpoint']
            
            logPath = config['logPath']
            run_tools.makedir_if_not_exists(logPath)
            logFile = os.path.join(logPath, '{}.json'.format(runName))
            print('log file: {}'.format(logFile))
            
            with open(logFile, 'w', encoding='utf-8') as f:
                json.dump(logData, f, ensure_ascii=False, indent=2, sort_keys=True, cls=NpEncoder)
            
            # pprint(logData)
            print('done.')
            gc.collect()
