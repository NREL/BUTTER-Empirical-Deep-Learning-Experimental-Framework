from abc import abstractmethod


class Preprocessor:
    
    @abstractmethod
    def forward(self, element):
        pass
    
    @abstractmethod
    def backward(self, element):
        pass
