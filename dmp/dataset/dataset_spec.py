from dataclasses import dataclass


@dataclass
class DatasetSpec():
    name: str  # migrate from dataset
    method: str  # migrate from test_split_method
    test_split: float  # direct migrate
    validation_split: float  # 0.0 when migrating from AspectTestTask
    label_noise: float  # direct migrate from label_noise is None or label_noise == 'none' or label_noise <= 0.0:
