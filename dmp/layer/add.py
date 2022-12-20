from typing import List,Union
from dmp.layer.a_element_wise_operator_layer import AElementWiseOperatorLayer
from dmp.layer.layer import Layer, register_layer_type

class Add(AElementWiseOperatorLayer):

    @staticmethod
    def make(input: List[Layer]) -> 'Add':
        return Add({}, input)

register_layer_type(Add)