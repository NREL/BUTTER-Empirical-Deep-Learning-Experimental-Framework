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
    
    for i,l in enumerate(model.layers):
        if i in config.keys():
            new_model.add(keras.layers.Dense(l.output_shape[1] + config[i], activation=l.activation))
        else:
            new_model.add(keras.layers.Dense(l.output_shape[1], activation=l.activation))
    
    new_model.build(model.input_shape)

    for i,l in enumerate(model.layers):
        old_weights = l.get_weights()
        new_weights = new_model.layers[i].get_weights()
        new_weights[0],new_weights[1] = scale * new_weights[0], scale * new_weights[1]

        new_weights[0][:old_weights[0].shape[0],:old_weights[0].shape[1]] = old_weights[0]
        new_weights[1][:old_weights[1].shape[0]] = old_weights[1]
        
        new_model.layers[i].set_weights(new_weights)
            
    del model

    return new_model


class AdditionalValidationSets(keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        Parameters:
        ----------
        validation_sets:
            a list of 3-tuples (validation_data, validation_targets, validation_set_name)
            or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        verbose:
            verbosity mode, 1 or 0
        batch_size:
            batch size to be used when evaluating on the additional datasets

        Source : https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [3, 4]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # # record the same values as History() as well
        # for k, v in logs.items():
        #     self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for metric, result in zip(self.model.metrics_names,results):
                valuename = validation_set_name + '_' + metric
                self.history.setdefault(valuename, []).append(result)