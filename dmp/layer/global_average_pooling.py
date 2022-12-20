from dmp.layer.global_pooling_layer import AGlobalPoolingLayer
from dmp.layer.layer import register_layer_type

class GlobalAveragePooling(AGlobalPoolingLayer):
    pass


register_layer_type(GlobalAveragePooling)