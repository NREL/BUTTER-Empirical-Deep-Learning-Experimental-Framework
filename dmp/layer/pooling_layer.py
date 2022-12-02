from typing import Tuple
from dmp.layer.spatitial_layer import ASpatitialLayer
from dmp.layer.layer import network_module_types


class APoolingLayer(ASpatitialLayer):

    @property
    def strides(self) -> Tuple:
        config = self.config
        strides = config.get('strides', None)
        if strides is not None:
            return strides
        return config['pool_size']


class MaxPool(APoolingLayer):
    pass


network_module_types.append(MaxPool)


class AvgPool(APoolingLayer):
    pass


network_module_types.append(AvgPool)


class AGlobalPoolingLayer(ASpatitialLayer):
    pass


class GlobalAveragePooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalAveragePooling)


class GlobalMaxPooling(AGlobalPoolingLayer):
    pass


network_module_types.append(GlobalMaxPooling)
