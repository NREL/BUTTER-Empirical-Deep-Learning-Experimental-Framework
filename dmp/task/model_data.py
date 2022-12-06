from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional
import tensorflow.keras as keras
import tensorflow

from dmp.layer.layer import Layer
from dmp.layer.visitor.keras_interface.layer_to_keras import KerasLayer


@dataclass
class ModelData():
    structure: Layer
    layer_shapes: Dict[Layer, Tuple]
    widths: List[int]
    num_free_parameters: int
    output_activation: str
    layer_to_keras_map: Dict[Layer, Tuple[KerasLayer, tensorflow.Tensor]]
    keras_model: keras.Model