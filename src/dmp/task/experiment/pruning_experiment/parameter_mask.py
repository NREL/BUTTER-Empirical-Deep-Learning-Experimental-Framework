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
        # print(f"mask call! {w.shape} {self.mask.shape}")
        if self.mask is None:
            self.mask = tensorflow.Variable(
                tensorflow.ones(w.shape, dtype=tensorflow.bool),
                shape=w.shape,
                trainable=False,
                dtype=tensorflow.bool,
            )

        return tensorflow.where(self.mask, w, 0)
