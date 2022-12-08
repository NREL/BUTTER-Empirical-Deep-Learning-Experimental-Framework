from dataclasses import dataclass
from dataclasses import dataclass

from dmp.layer.layer import Layer
from dmp.model.keras_layer_info import KerasLayerInfo


@dataclass
class LayerGrowthInfo:
    src: KerasLayerInfo
    dest: KerasLayerInfo
