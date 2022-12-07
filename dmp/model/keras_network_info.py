from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from dmp.model.keras_layer_info import KerasLayerInfo, KerasLayer
from dmp.layer import Layer


@dataclass
class KerasNetworkInfo():
    layer_to_keras_map: Dict[Layer, KerasLayerInfo]
    inputs : List[KerasLayer]
    outputs : List[KerasLayer]
    
