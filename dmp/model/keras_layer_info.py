from dataclasses import dataclass, field
from typing import Sequence, Any, Tuple, Dict, Union

import tensorflow.keras as keras
import tensorflow

from dmp.layer.layer import Layer

KerasLayer = Union[keras.layers.Layer, tensorflow.Tensor]


@dataclass
class KerasLayerInfo():
    layer: Layer
    keras_layer: KerasLayer
    output_tensor: tensorflow.Tensor
