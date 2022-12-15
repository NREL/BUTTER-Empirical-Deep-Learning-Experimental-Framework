from typing import List,Union
from dmp.layer.a_element_wise_operator_layer import AElementWiseOperatorLayer
from dmp.layer.layer import Layer, network_module_types

class Add(AElementWiseOperatorLayer):

    @staticmethod
    def make(input: List[Layer]) -> 'Add':
        return Add({}, input)

network_module_types.append(Add)