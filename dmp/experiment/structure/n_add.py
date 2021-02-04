from dmp.experiment.structure.network_module import NetworkModule


class NAdd(NetworkModule):

    def __init__(self,
                 inputs: ('NetworkModule', ...),
                 ) -> None:
        super().__init__(inputs, inputs[0].shape)
