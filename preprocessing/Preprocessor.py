from abc import abstractmethod
from typing import (
    Iterable,
    Union,
    )


class Preprocessor:
    
    @abstractmethod
    def forward(self, element: any) -> any:
        pass
    
    @abstractmethod
    def backward(self, element: any) -> any:
        pass
