from dmp.layer.global_pooling_layer import GlobalPoolingLayer
from dmp.layer.layer import register_layer_type

class GlobalAveragePooling(GlobalPoolingLayer):
    pass


register_layer_type(GlobalAveragePooling)