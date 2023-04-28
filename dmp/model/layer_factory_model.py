from dataclasses import dataclass
from typing import Optional
from dmp.layer.layer import Layer, LayerFactory
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo


class LayerFactoryModel(ModelSpec):
    '''
    Adaptor class that allows a LayerFactory to act as a ModelSpec
    '''

    def __init__(
        self,
        layer_factory: LayerFactory,
        input: Optional[Layer] = None,  # set to None for runtime determination
        output: Optional[Layer] = None,  # set to None for runtime determination
    ) -> None:
        super().__init__(
            input=input,
            output=output,
        )
        self.layer_factory: LayerFactory = layer_factory

    def make_network(self) -> NetworkInfo:
        return NetworkInfo(
            self.output.make_layer(  # type: ignore
                {},
                self.layer_factory.make_layer({}, self.input),  # type: ignore
            ),
            {},
        )
