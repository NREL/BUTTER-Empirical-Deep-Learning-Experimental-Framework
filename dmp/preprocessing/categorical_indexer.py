from typing import Iterable

from dmp.preprocessing.preprocessor import Preprocessor


class CategoricalIndexer(Preprocessor):
    
    def __init__(self, data: Iterable):
        forward_mapping = {}
        backward_mapping = []
        for element in data:
            if element in forward_mapping:
                continue
            forward_mapping[element] = len(backward_mapping)
            backward_mapping.append(element)
        self._forward_mapping = forward_mapping
        self._backward_mapping = backward_mapping
    
    def forward(self, element) -> int:
        return self._forward_mapping[element]
    
    def backward(self, element: int):
        return self._backward_mapping[element]
    
    @property
    def index_size(self) -> int:
        return len(self._backward_mapping)
