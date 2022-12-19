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
    _prepare_inputs_function: Callable[['FunctionalPMLBDatasetLoader', Any], Any]

    def _prepare_inputs(self, data):
        return self._prepare_inputs_function(self, data)
