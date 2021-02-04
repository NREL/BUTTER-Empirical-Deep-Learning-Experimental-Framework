from dmp.experiment.structure.network_module import NetworkModule


class NInput(NetworkModule):

    def __init__(self,
                 inputs: ('NetworkModule', ...),
                 shape: (int, ...),
                 ) -> None:
        super().__init__(inputs, shape)
