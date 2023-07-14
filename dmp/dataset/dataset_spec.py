from dataclasses import dataclass


@dataclass
class DatasetSpec:
    """
    Defines a dataset to use in a TrainingExperiment
    """

    name: str  # name of the dataset
    source: str  # source of the dataset
    method: str  # how to split the validation and test data
    test_split: float  # % test split from dataset
    validation_split: float  # % validation split from dataset
    label_noise: float  # % label noise to add to training data
