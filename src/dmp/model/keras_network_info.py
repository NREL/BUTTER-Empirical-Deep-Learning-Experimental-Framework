from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from dmp.model.keras_layer_info import KerasLayerInfo, KerasLayer
from dmp.layer import Layer


@dataclass
class KerasNetworkInfo:
    """
    Details about a keras network made from a NetworkInfo
    """

    layer_to_keras_map: Dict[
        Layer, KerasLayerInfo
    ]  # mapping between Layer instances and KerasLayerInfo's
    inputs: List[KerasLayer]  # inputs to the keras model
    outputs: List[KerasLayer]  # outputs to the keras model
