from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from dataclasses import dataclass
from dmp.dataset.ml_task import MLTask
from dmp.dataset.pmlb_dataset_loader import PMLBDatasetLoader


class FunctionalPMLBDatasetLoader(PMLBDatasetLoader):
    _prepare_inputs_function: Callable[["FunctionalPMLBDatasetLoader", Any], Any]

    def __init__(
        self,
        dataset_name: str,
        ml_task: MLTask,
        prepare_inputs_function: Callable[["FunctionalPMLBDatasetLoader", Any], Any],
    ):
        super().__init__(dataset_name, ml_task)
        self._prepare_inputs_function = prepare_inputs_function

    def _prepare_inputs(self, data):
        return self._prepare_inputs_function(self, data)
