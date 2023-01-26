from typing import Sequence


class TrainingExperimentKeys():

    epoch: str = 'epoch'
    test: str = 'test'
    train: str = 'train'
    trained: str = 'trained'
    validation: str = 'validation'

    train_start_timestamp: str = 'train_start_timestamp'
    
    interval_suffix : str = '_ms'
    epoch_start_time_ms: str = 'epoch_start_ms'
    epoch_time_ms: str = 'train_ms'

    epoch_end_key: str = 'relative_test_time'
    epoch_end_key: str = 'relative_test_start_time'
    epoch_end_key: str = 'relative_test_time'

    extended_history_columns: Sequence[str] = (
        'cosine_similarity',
        'kullback_leibler_divergence',
        'root_mean_squared_error',
        'mean_absolute_error',
        'mean_squared_logarithmic_error',
        'hinge',
        'squared_hinge',
        'categorical_hinge',
    )
    
