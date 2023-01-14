from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from dataclasses import dataclass
from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset_loader import DatasetLoader

import multiprocess
__lock = multiprocess.Lock()

@dataclass
class PMLBDatasetLoader(DatasetLoader):

    def _fetch_from_source(self):
        with __lock:
            import pmlb
            
            return Dataset(self.ml_task,
                        DatasetGroup(*pmlb.fetch_data(
                            self.dataset_name,
                            return_X_y=True,
                        )))  # type: ignore
