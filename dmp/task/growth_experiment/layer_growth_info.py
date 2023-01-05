from dataclasses import dataclass
from dataclasses import dataclass
from typing import Optional

from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo


@dataclass
class LayerGrowthInfo:
    src: Optional[KerasLayerInfo]
    dst: Optional[KerasLayerInfo]
