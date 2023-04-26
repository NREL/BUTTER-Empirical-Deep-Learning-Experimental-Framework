from typing import Any, Callable, List, Optional, Tuple, Dict
from dmp.model.factory_model import FactoryModel
from dmp.layer import *


class SequentialModel(FactoryModel):
    '''
    LayerFactory and ModelSpec that defines a sequence of serially connected layers.
    '''

    def __init__(
        self,
        layer_factories: List[LayerFactory], # layers from input to output 
        input: Optional[Layer] = None,
        output: Optional[Layer] = None,
    ) -> None:
        super().__init__(input, output)
        self.layer_factories: List[LayerFactory] = layer_factories

    def make_layer(
        self,
        inputs: Union['Layer', List['Layer']],
        config: 'LayerConfig',
    ) -> Layer:
        layer_input = inputs
        for layer_factory in self.layer_factories:
            layer_input = layer_factory.make_layer(layer_input, config)
        if isinstance(layer_input, List):
            layer_input = layer_input[0]
        return layer_input
