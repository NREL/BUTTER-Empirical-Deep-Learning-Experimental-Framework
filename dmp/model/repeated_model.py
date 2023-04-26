from typing import Any, Callable, List, Optional, Tuple, Dict
from dmp.layer import *
from dmp.model.factory_model import FactoryModel
from dmp.model.sequential_model import SequentialModel


class RepeatedModel(FactoryModel):
    '''
    LayerFactory and ModelSpec that defines a repeated sequence of identical, serially connected layers.
    '''

    def __init__(
        self,
        layer: Layer,
        depth: int,
        input: Optional[Layer] = None,
        output: Optional[Layer] = None,
    ) -> None:
        super().__init__(input, output)
        self.layer: Layer = layer
        self.depth: int = depth

    def make_layer(
        self,
        inputs: Union['Layer', List['Layer']],
        config: 'LayerConfig',
    ) -> Layer:
        return SequentialModel(
            [self.layer] * self.depth, self.input, self.output
        ).make_layer(inputs, config)

def repeat(layer:Layer, times:int)->RepeatedModel:
    return 