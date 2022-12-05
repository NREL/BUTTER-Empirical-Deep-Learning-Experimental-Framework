from hashlib import new
import numpy as np
from tensorflow import keras


def grow_network(model, config, scale=0):
    """
    This function grows the widths of a given MLP model. Old to old weights remain the same.
    Old to new and new to new weights get randomly initialized. 
    New to old weights get scaled by 'scale' parameter.
    
    Parameters
    ----------
    model (keras.Sequential) : model to grow
    config (dict) : dictionary of what layers to grow. Ex. {0:8, 1:12} adds 8 neurons to first hidden layer
        and 12 to second
    scale (float) : how much to scale down the weights from new neurons to old neurons

    Returns
    -------
    new_model (keras.Sequential) : model after growth
    """
    new_model = keras.Sequential()
    print([l.output_shape for l in model.layers[1:]])
    input_layer_exists = False
    for i, l in enumerate(model.layers):
        if type(l) == keras.layers.InputLayer:
            input_layer_exists = True
            continue
        elif input_layer_exists:  # if an input layer exists then our config is off by one
            i = i - 1

        if i in config.keys():
            new_model.add(
                keras.layers.Dense(l.output_shape[1] + config[i],
                                   activation=l.activation))
        else:
            new_model.add(
                keras.layers.Dense(l.output_shape[1], activation=l.activation))

    new_model.build(model.input_shape)
    print([l.output_shape for l in new_model.layers])

    input_layer_exists = False
    for i, l in enumerate(model.layers):
        if type(l) == keras.layers.InputLayer:
            input_layer_exists = True
            continue
        elif input_layer_exists:  # if an input layer exists then our config is off by one
            i = i - 1

        old_weights = l.get_weights()
        new_weights = new_model.layers[i].get_weights()
        new_weights[0], new_weights[
            1] = scale * new_weights[0], scale * new_weights[1]

        new_weights[0][:old_weights[0].shape[0], :old_weights[0].
                       shape[1]] = old_weights[0]
        new_weights[1][:old_weights[1].shape[0]] = old_weights[1]

        new_model.layers[i].set_weights(new_weights)

    del model

    return new_model


