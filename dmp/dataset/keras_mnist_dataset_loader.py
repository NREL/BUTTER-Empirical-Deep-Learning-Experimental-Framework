from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

from dmp.dataset.keras_image_dataset_loader import KerasImageDatasetLoader


class KerasMNISTDatasetLoader(KerasImageDatasetLoader):

    def __init__(
        self,
        dataset_name: str,
        keras_load_data_function: Callable,
    ) -> None:
        super().__init__(dataset_name, keras_load_data_function)

    def _prepare_inputs(self, data):
        return super()._prepare_inputs(data.reshape(data.shape[0], 28, 28, 1))