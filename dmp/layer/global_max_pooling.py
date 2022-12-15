from dmp.layer.layer import network_module_types
from dmp.layer.global_average_pooling import AGlobalPoolingLayer


class GlobalMaxPooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalMaxPooling)
