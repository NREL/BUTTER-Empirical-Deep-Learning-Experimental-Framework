from abc import abstractmethod


class Preprocessor:
    
    @abstractmethod
    def forward(self, element: any) -> any:
        pass
    
    @abstractmethod
    def backward(self, element: any) -> any:
        pass
