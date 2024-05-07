from typing import Iterable

import numpy

from dmp.preprocessing.categorical_indexer import CategoricalIndexer


class OneHotIndexer(CategoricalIndexer):
    def __init__(self, data: Iterable):
        super().__init__(data)

    def forward(self, element) -> int:
        result = numpy.zeros(self.index_size)
        result[super().forward(element)] = 1
        return result  # type: ignore

    def backward(self, element: int):
        return self._backward_mapping[element]

    @property
    def index_size(self) -> int:
        return super().index_size
