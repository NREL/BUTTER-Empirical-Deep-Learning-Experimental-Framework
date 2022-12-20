from dmp.layer.layer import register_layer_type
from dmp.layer.global_average_pooling import AGlobalPoolingLayer


class GlobalMaxPooling(AGlobalPoolingLayer):
    pass


register_layer_type(GlobalMaxPooling)
