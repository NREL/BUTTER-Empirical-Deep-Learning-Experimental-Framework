from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

from dmp.dataset.keras_dataset_loader import KerasDatasetLoader


class KerasMNISTDatasetLoader(KerasDatasetLoader):

    def __init__(
        self,
        dataset_name: str,
        keras_load_data_function: Callable,
    ) -> None:
        super().__init__(dataset_name, keras_load_data_function)

    def _prepare(self, data):
        raw_inputs, raw_outputs = data
        raw_inputs = raw_inputs.reshape(raw_inputs.shape[0], 28, 28, 1)
        return super()._prepare((raw_inputs, raw_outputs))