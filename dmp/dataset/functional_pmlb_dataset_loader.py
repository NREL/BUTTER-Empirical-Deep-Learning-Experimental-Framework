from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from dataclasses import dataclass
from dmp.dataset.pmlb_dataset_loader import PMLBDatasetLoader


@dataclass
class FunctionalPMLBDatasetLoader(PMLBDatasetLoader):
    _prepare_function: Callable[['FunctionalPMLBDatasetLoader', Any], Any]

    def _prepare_dataset_data(self, data):
        return self._prepare_function(self, data)
