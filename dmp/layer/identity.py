from typing import List
from dmp.layer.layer import Layer, LayerConfig, register_layer_type
from dmp.layer.a_element_wise_operator_layer import AElementWiseOperatorLayer


class Identity(AElementWiseOperatorLayer):

    def make_layer(
        self,
        inputs: List[Layer],
        override_if_exists: LayerConfig,
    ) -> Layer:
        # try to elide the identity layer

        if len(self.inputs) >= 1:
            return self.input.make_layer(inputs, override_if_exists)

        if len(inputs) >= 1:
            return inputs[0]

        return self


register_layer_type(Identity)