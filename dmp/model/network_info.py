from typing import Optional, Sequence, Any, Tuple, Dict

from dmp.layer.layer import Layer
from dmp.layer.visitor.compute_layer_shapes import compute_layer_shapes
from dmp.layer.visitor.count_free_parameters import count_free_parameters


class NetworkInfo():

    def __init__(self, structure: Layer, description: Dict[str, Any]) -> None:
        self.structure :Layer= structure
        self.description:Dict[str,Any] = description
        compute_layer_shapes(structure)
        self.num_free_parameters :int = count_free_parameters(structure)
    
