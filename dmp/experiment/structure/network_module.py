from dmp.experiment.structure.module_type import ModuleType


class NetworkModule:
    def __init__(self, type: ModuleType, inputs: ('NetworkModule', ...), shape: (int, ...),
                 activation: any = None) -> None:
        self.type = type
        self.inputs: (NetworkModule, ...) = inputs
        self.shape: (int, ...) = shape
        self.activation = activation

    @property
    def size(self) -> int:
        acc = 1
        for dim in self.shape:
            acc *= dim
        return acc
