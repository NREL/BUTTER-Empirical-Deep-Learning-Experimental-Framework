import tensorflow


class WeightMask(tensorflow.keras.constraints.Constraint):
    """
    A class that implements a constraint function to mask weights in a neural network model.
    """

    def __init__(
        self,
        mask_group: int = 0,
    ):
        super().__init__()
        self.mask_group: int = mask_group
        self.mask = tensorflow.Variable(
            True,
            shape=tensorflow.TensorShape(None),
            trainable=False,
            dtype=tensorflow.bool,
        )

    def __call__(self, w):
        """
        Used by tensorflow to add the constraint to the computation graph.
        """
        return tensorflow.where(self.mask, w, 0)
