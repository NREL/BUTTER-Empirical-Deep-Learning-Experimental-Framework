from typing import Iterable

import numpy

from preprocessing.Preprocessor import Preprocessor


class Normalizer(Preprocessor):
    
    def __init__(self, data: Iterable):
        # Welford's algorithm
        n = 0
        mean = 0.0
        M2 = 0.0
        for element in data:
            n += 1
            delta = element - mean
            mean += delta / n
            delta2 = element - mean
            M2 += delta * delta2
        
        self.mean = mean
        self.standardDeviation = numpy.sqrt(M2 / n)
        # self.sampleVariance = M2 / (n - 1)
    
    def forward(self, element: any) -> any:
        return (element - self.mean) / self.standardDeviation
    
    def backward(self, element: any) -> any:
        return (element * self.standardDeviation) + self.mean
