from typing import List
from dmp.layer.layer import Layer, LayerConfig
from dmp.layer.element_wise_operator_layer import ElementWiseOperatorLayer


class Identity(ElementWiseOperatorLayer):
    pass

    # def make_layer(
    #     self,
    #     inputs: List[Layer],
    #     override_if_exists: LayerConfig,
    # ) -> Layer:
    #     # try to elide the identity layer

    #     if len(self.inputs) >= 1:
    #         return self.input.make_layer(inputs)

    #     if len(inputs) >= 1:
    #         return inputs[0]

    #     return self

