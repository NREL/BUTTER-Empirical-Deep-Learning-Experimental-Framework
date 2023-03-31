from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WeightMask(tf.keras.constraints.Constraint):
    """
    A class that implements a constraint function to mask weights in a neural network model.
    """

    def __init__(
        self,
        mask_group: int = 0,
    ):
        super().__init__()
        logger.debug("__init__")
        self.mask_group: int = mask_group
        self.mask = tf.Variable(
            True, shape=tf.TensorShape(None), trainable=False, dtype=tf.bool
        )

    def __call__(self, w):
        """
        Used by tensorflow to add the constraint to the computation graph.
        """
        logger.debug("__call__")
        return tf.where(self.mask, w, 0)
