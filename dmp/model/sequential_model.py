from typing import Any, Callable, List, Optional, Tuple, Dict
from dmp.model.factory_model import FactoryModel
from dmp.layer import *


class SequentialModel(FactoryModel):
    '''
    LayerFactory and ModelSpec that defines a sequence of serially connected layers.
    '''

    def __init__(
        self,
        layers: List[Layer], # layers from input to output 
        input: Optional[Layer] = None,
        output: Optional[Layer] = None,
    ) -> None:
        super().__init__(input, output)
        self.layers: List[Layer] = layers

    def make_layer(
        self,
        inputs: List[Layer],
        config: 'LayerConfig',
    ) -> Layer:
        previous_layers = inputs
        for layer in self.layers:
            previous_layers = [layer.make_layer(previous_layers, config)]
        return previous_layers[0]
