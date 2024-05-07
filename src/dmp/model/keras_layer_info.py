from dataclasses import dataclass, field
from typing import Sequence, Any, Tuple, Dict, Union

import tensorflow.keras as keras
import tensorflow

from dmp.layer.layer import Layer

KerasLayer = Union[keras.layers.Layer, tensorflow.Tensor]


@dataclass
class KerasLayerInfo:
    """
    Mapping information between a Layer and keras.Layer instance
    """

    layer: Layer  # a Layer instance
    keras_layer: KerasLayer  # the keras layer corresponding to this Layer
    output_tensor: tensorflow.Tensor  # output tensor of the layer
