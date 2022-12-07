from dataclasses import dataclass, field
from typing import Sequence, Any, Tuple, Dict, Union

import tensorflow.keras as keras
import tensorflow

KerasLayer = Union[keras.layers.Layer, tensorflow.Tensor]

@dataclass
class KerasLayerInfo():
    keras_layer : KerasLayer
    output_tensor : tensorflow.Tensor
