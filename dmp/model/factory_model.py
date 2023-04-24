from typing import Optional
from dmp.layer.layer import Layer, LayerFactory
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo


class FactoryModel(LayerFactory, ModelSpec):
    '''
    Abstract class for classes that are both LayerFactory's and ModelSpec's
    '''

    def __init__(
        self,
        input: Optional[Layer] = None,
        output: Optional[Layer] = None,
    ) -> None:
        super().__init__(input, output)

    def make_network(self) -> NetworkInfo:
        return NetworkInfo(
            self.output.make_layer(  # type: ignore
                [
                    self.make_layer([self.input], {}),  # type: ignore
                ],
                {},
            ),
            {},
        )
