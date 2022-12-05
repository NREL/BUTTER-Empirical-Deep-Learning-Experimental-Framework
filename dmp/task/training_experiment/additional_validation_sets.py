import keras


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
            if len(validation_set) not in [2, 3, 4]:
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
            if len(validation_set) == 2:
                validation_history_key, validation_data = validation_set
                validation_targets = None
                sample_weights = None
            if len(validation_set) == 3:
                validation_history_key, validation_data, validation_targets = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_history_key, validation_data, validation_targets, sample_weights = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for metric, result in zip(self.model.metrics_names, results):
                valuename = validation_history_key + '_' + metric
                self.history.setdefault(valuename, []).append(result)