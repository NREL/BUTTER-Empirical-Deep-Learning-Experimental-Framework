from abc import ABC, abstractmethod
from typing import Dict
from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo


class PruningMethod(ABC):
    '''
    Prunes a layer and its descendants in some way.
    '''

    @abstractmethod
    def prune(
        self,
        root: Layer,
        layer_to_keras_map: Dict[Layer, KerasLayerInfo],
    ) -> int: # returns number of weights pruned
        pass