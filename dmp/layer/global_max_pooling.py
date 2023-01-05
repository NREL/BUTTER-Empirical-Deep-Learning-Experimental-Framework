from dmp.layer.layer import register_layer_type
from dmp.layer.global_average_pooling import GlobalPoolingLayer


class GlobalMaxPooling(GlobalPoolingLayer):
    pass


register_layer_type(GlobalMaxPooling)
