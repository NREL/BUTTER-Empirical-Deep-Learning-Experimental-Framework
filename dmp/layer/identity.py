from dmp.layer.layer import Layer, network_module_types
from dmp.layer.a_element_wise_operator_layer import AElementWiseOperatorLayer

class Identity(AElementWiseOperatorLayer):
    pass


network_module_types.append(Identity)