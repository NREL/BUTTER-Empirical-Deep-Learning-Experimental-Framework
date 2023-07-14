from typing import List, Union
from dmp.layer.element_wise_operator_layer import ElementWiseOperatorLayer
from dmp.layer.layer import Layer


class Add(ElementWiseOperatorLayer):
    @staticmethod
    def make(input: Union["Layer", List["Layer"]]) -> "Add":
        return Add({}, input)
