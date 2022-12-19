from dmp.layer.global_pooling_layer import AGlobalPoolingLayer
from dmp.layer.layer import network_module_types

class GlobalAveragePooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalAveragePooling)