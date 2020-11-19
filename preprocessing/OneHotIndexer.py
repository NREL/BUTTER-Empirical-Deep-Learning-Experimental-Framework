from typing import Iterable

import numpy

from preprocessing.CategoricalIndexer import CategoricalIndexer
from preprocessing.Preprocessor import Preprocessor


class OneHotIndexer(CategoricalIndexer):
    
    def __init__(self, data: Iterable):
        super().__init__(data)
    
    def forward(self, element: any) -> int:
        result = numpy.zeros(self.indexSize)
        result[super().forward(element)] = 1
        return result
    
    def backward(self, element: int) -> any:
        return self._backwardMapping[element]
    
    @property
    def indexSize(self) -> int:
        return self._categoricalIndexer.indexSize
