from dmp.layer.a_element_wise_operator_layer import AElementWiseOperatorLayer
from dmp.layer.layer import Layer, network_module_types

class Zeroize(AElementWiseOperatorLayer):
    pass


network_module_types.append(Zeroize)
