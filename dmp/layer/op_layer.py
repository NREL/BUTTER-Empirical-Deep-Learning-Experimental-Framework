from typing import List, Union
from dmp.keras_interface.keras_utils import make_keras_config
from dmp.layer.element_wise_operator_layer import ElementWiseOperatorLayer
from dmp.layer.layer import Layer, LayerConfig, empty_config, empty_inputs


class OpLayer(Layer):
    """
    Layer that applies a keras operator to the inputs.
    The layer config is the keras config as defined in keras_utils.py.
    """

    @staticmethod
    def make(
        type_name: str,
        config: LayerConfig = empty_config,  # keras class constructor kwargs
        inputs: Union["Layer", List["Layer"]] = empty_inputs,  # inputs to this Layer
    ) -> "OpLayer":
        config = make_keras_config(type_name, config)
        return OpLayer(config, inputs)
