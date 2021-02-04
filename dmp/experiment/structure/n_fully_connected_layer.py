from dmp.experiment.structure.network_module import NetworkModule


class NFullyConnectedLayer(NetworkModule):

    def __init__(self,
                 inputs: ('NetworkModule', ...),
                 shape: (int, ...),
                 activation: any,
                 ) -> None:
        super().__init__(inputs, shape)
        self.activation = activation
