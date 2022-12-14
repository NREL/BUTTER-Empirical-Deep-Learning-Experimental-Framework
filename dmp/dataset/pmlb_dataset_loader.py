from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from dataclasses import dataclass
import pmlb
from dmp.dataset.dataset_loader import DatasetLoader


@dataclass
class PMLBDatasetLoader(DatasetLoader):

    def _fetch_from_source(self):
        return pmlb.fetch_data(self.dataset_name, return_X_y=True)


