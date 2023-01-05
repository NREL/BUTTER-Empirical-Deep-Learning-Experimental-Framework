from dmp.layer.element_wise_operator_layer import ElementWiseOperatorLayer
from dmp.layer.layer import register_layer_type

class Zeroize(ElementWiseOperatorLayer):
    pass


register_layer_type(Zeroize)
