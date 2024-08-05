from abc import ABC, abstractmethod
from typing import List

import numpy

from dmp.layer.layer import Layer


class PruningEvaluator(ABC):

    @abstractmethod
    def compute_pruning_values(
        self,
        root: Layer,
        prunable_layers: List[Layer],
        prunable_weights: numpy.ndarray,
    ) -> numpy.ndarray:
        pass
