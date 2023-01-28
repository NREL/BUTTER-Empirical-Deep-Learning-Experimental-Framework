from typing import List, Sequence, Set


class TrainingExperimentKeys():

    def __init__(self) -> None:
        self.run: str = 'run'
        self.epoch: str = 'epoch'
        self.count:str = 'count'
        self.test: str = 'test'
        self.train: str = 'train'
        self.trained: str = 'trained'
        self.validation: str = 'validation'

        self.data_set_prefixes: Set[str] = {
            p + '_'
            for p in [
                'test',
                'train',
                'validation',
            ]
        }

        self.train_start_timestamp: str = 'train_start_timestamp'

        self.interval_suffix: str = '_ms'
        self.epoch_start_time_ms: str = 'epoch_start_ms'
        self.epoch_time_ms: str = 'train_ms'

        self.epoch_end_key: str = 'relative_test_time'
        self.epoch_end_key: str = 'relative_test_start_time'
        self.epoch_end_key: str = 'relative_test_time'

        self.simple_summarize_keys: Set[str] = set([
            self.epoch_time_ms,
        ] + [p + self.interval_suffix for p in self.data_set_prefixes])

        self.extended_history_columns: Set[str] = {
            'cosine_similarity',
            'kullback_leibler_divergence',
            'root_mean_squared_error',
            'mean_absolute_error',
            'mean_squared_logarithmic_error',
            'hinge',
            'squared_hinge',
            'categorical_hinge',
        }

        self.loss_metrics: Set[str] = {
            'test_loss',
            'validation_loss',
            'categorical_crossentropy',
            'mean_squared_error',
            'binary_crossentropy',
        }


keys = TrainingExperimentKeys()