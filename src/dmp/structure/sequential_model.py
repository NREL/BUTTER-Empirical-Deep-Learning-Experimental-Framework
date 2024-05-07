from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Dict, Union
from dmp.layer.layer import Layer, LayerConfig, LayerFactory


@dataclass
class SequentialModel(LayerFactory):
    """
    Defines a sequence of serially connected layers.
    """

    layer_factories: List[LayerFactory]  # layers from input to output

    def make_layer(
        self,
        config: "LayerConfig",
        inputs: Union["Layer", List["Layer"]],
    ) -> Layer:
        layer_input = inputs
        for layer_factory in self.layer_factories:
            layer_input = layer_factory.make_layer(config, layer_input)
        if isinstance(layer_input, List):
            layer_input = layer_input[0]
        return layer_input
