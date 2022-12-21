import tensorflow.keras as keras


class ValidationHistoryRecorder(keras.callbacks.Callback):

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
        super(ValidationHistoryRecorder, self).__init__()

        self.validation_sets = []
        for validation_set in validation_sets:
            validation_history_key = None
            validation_data = None
            validation_targets = None
            sample_weights = None

            if len(validation_set) == 2:
                validation_history_key, validation_data = validation_set
                validation_targets = None
                sample_weights = None
            elif len(validation_set) == 3:
                validation_history_key, validation_data, validation_targets = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_history_key, validation_data, validation_targets, sample_weights = validation_set
            else:
                raise ValueError()
            
            self.validation_sets.append((
                validation_history_key,
                validation_data,
                validation_targets,
                sample_weights,
            ))

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

        model: keras.Model = self.model  # type: ignore

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            (
                validation_history_key,
                validation_data,
                validation_targets,
                sample_weights,
            ) = validation_set

            results = model.evaluate(
                x=validation_data,  # type: ignore
                y=validation_targets,
                verbose=self.verbose,
                sample_weight=sample_weights,
                batch_size=self.batch_size,
            )

            for metric, result in zip(model.metrics_names, results):
                valuename = validation_history_key + '_' + metric
                self.history.setdefault(valuename, []).append(result)
