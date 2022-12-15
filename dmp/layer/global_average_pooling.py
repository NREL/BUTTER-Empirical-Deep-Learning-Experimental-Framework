from dmp.layer.layer import network_module_types
from dmp.layer.global_average_pooling import AGlobalPoolingLayer


class GlobalAveragePooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalAveragePooling)