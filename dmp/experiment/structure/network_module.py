

class NetworkModule:
    def __init__(self,
                 inputs: ('NetworkModule', ...),
                 shape: (int, ...),
                 ) -> None:
        self.inputs: (NetworkModule, ...) = inputs
        self.shape: (int, ...) = shape

    @property
    def size(self) -> int:
        acc = 1
        for dim in self.shape:
            acc *= dim
        return acc
