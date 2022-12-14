from dataclasses import dataclass
from typing import Dict
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.ml_task import MLTask


@dataclass
class DatasetData:
    ml_task: MLTask
    splits: Dict[str, DatasetGroup]
