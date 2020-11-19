from typing import Iterable

from preprocessing.Preprocessor import Preprocessor


class CategoricalIndexer(Preprocessor):
    
    def __init__(self, data: Iterable):
        forwardMapping = {}
        backwardMapping = []
        for element in data:
            if element in forwardMapping:
                continue
            forwardMapping[element] = len(backwardMapping)
            backwardMapping.append(element)
        self._forwardMapping = forwardMapping
        self._backwardMapping = backwardMapping
    
    def forward(self, element: any) -> int:
        return self._forwardMapping[element]
    
    def backward(self, element: int) -> any:
        return self._backwardMapping[element]
    
    @property
    def indexSize(self) -> int:
        return len(self._backwardMapping)
