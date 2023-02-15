from dataclasses import dataclass
import os
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)

import numpy
from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset_loader import DatasetLoader, dataset_cache_directory
from dmp.dataset.ml_task import MLTask


@dataclass
class ImageNetDatasetLoader(DatasetLoader):

    size: int
    crop: Optional[int]

    def __init__(
        self,
        dataset_name: str,
        ml_task: MLTask,
        size: int,
        crop: Optional[int],
    ):
        super().__init__('imagenet', dataset_name, ml_task)
        self.size = size
        self.crop = crop

    def _fetch_from_source(self) -> Dataset:

        def make_group(raw_array):
            inputs = raw_array['data']
            outputs = raw_array['labels']
            outputs -= 1  # subtract 1 to make labels start at 0

            if self.crop is not None:
                inds = outputs < self.crop
                inputs = inputs[inds]
                outputs = outputs[inds]

            outputs = outputs.astype(int)
            n = outputs.shape[0]
            inputs = inputs.reshape(n, 3, self.size, self.size)
            inputs = numpy.transpose(inputs, (0, 2, 3, 1))
            return DatasetGroup(inputs, outputs)

        batches = 10
        arrays = []
        for batch in range(1, batches + 1):
            file_path = os.path.join(
                dataset_cache_directory,
                f'Imagenet{self.size}_train_npz',
                f'train_data_batch_{batch}.nzp',
            )
            arrays.append(numpy.load(file_path))
        concatenated = numpy.concatenate(arrays)
        del arrays

        return Dataset(
            self.ml_task,
            make_group(concatenated),
            make_group(
                numpy.load(
                    os.path.join(
                        dataset_cache_directory,
                        f'Imagenet{self.size}_val_npz',
                        'val_data.nzp',
                    ))),
        )

    def _prepare_inputs(self, data):
        return self._prepare_image(data)

    # def _fetch_from_source(self):
    #     # dataset_cache_directory
    #     # f'Imagenet{self.size}_train_npz/train_data_batch_{batch}.nzp'
    #     # f'Imagenet{self.size}_val_npz/val_data.nzp'

    #     batches = 10

    #     test_path = os.path.join(dataset_cache_directory, f'Imagenet{self.size}_val_npz', 'val_data.nzp')
    #     d = numpy.load(test_path)

    #     # print('loaded test data')
    #     raw_inputs = d['data']
    #     raw_outputs = d['labels']
    #     train_path = os.path.join(dataset_cache_directory, f'Imagenet{self.size}_train_npz')
    #     for batch in range(1, batches + 1):
    #         file_path = os.path.join(train_path, f'train_data_batch_{batch}.nzp')
    #         d = numpy.load(file_path)
    #         raw_inputs = numpy.concatenate((raw_inputs, d['data']), axis=0)
    #         raw_outputs = numpy.concatenate((raw_outputs, d['labels']), axis=0)
    #     raw_outputs -= 1  # subtract 1 to make labels start at 0

    #     if self.crop is not None:
    #         inds = raw_outputs < self.crop
    #         raw_inputs = raw_inputs[inds]
    #         raw_outputs = raw_outputs[inds]

    #     raw_outputs = raw_outputs.astype(int)
    #     n = raw_outputs.shape[0]
    #     raw_inputs = raw_inputs.reshape(n, 3, self.size, self.size)
    #     raw_inputs = numpy.transpose(raw_inputs, (0, 2, 3, 1))
    #     return raw_inputs, raw_outputs

    # def _prepare_dataset_data(self, data):
    #     return self._prepare_image(data)
