from dmp.layer.a_element_wise_operator_layer import AElementWiseOperatorLayer
from dmp.layer.layer import register_layer_type

class Zeroize(AElementWiseOperatorLayer):
    pass


register_layer_type(Zeroize)
