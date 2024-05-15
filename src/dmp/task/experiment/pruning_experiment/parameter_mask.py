import tensorflow


class ParameterMask(tensorflow.keras.constraints.Constraint):
    """
    A class that implements a constraint function to mask weights in a neural network model.
    """

    def __init__(
        self,
        mask_group: int = 0,
    ):
        super().__init__()
        self.mask_group: int = mask_group
        self.mask = None

    def __call__(self, w):
        """
        Used by tensorflow to add the constraint to the computation graph.
        """
        # print(f"mask call! {self.name} {w.shape} {self.mask.shape}")
        return tensorflow.where(self.get_mask(w.shape), w, 0)

    def get_mask(self, shape):
        if self.mask is None:
            self.mask = tensorflow.Variable(
                tensorflow.ones(shape, dtype=tensorflow.bool),
                shape=shape,
                trainable=False,
                dtype=tensorflow.bool,
            )
        return self.mask

    def set_mask(self, source):
        self.get_mask(source.shape).assign(source)
