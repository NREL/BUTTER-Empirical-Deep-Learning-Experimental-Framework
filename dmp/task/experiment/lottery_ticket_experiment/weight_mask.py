import tensorflow as tf
import numpy as np
import logging
logger = logging.getLogger(__name__)

class WeightMask(tf.keras.constraints.Constraint):
    """
    A class that implements a constraint function to mask weights in a neural network model.
    """
    def __init__(self):
        super().__init__()
        logger.debug("__init__")
        self.mask = tf.Variable(True, shape=tf.TensorShape(None), trainable=False, dtype=tf.bool)
        # I'd prefer to initialize it with the correct shape, we could pass in the kernel_size. But why don't we have access to it?
        # I'd also prefer to not have to initialize this to 1.0, but it is required. Maybe a placeholder variable could work?

    def updateMask(self, mask):
        """
        updateMask function.
        To try and get around the fact that __call__ is only called once at model inititialization, I use a single tensor Variable whose values are updated each iteration.
        """
        logger.debug("updateMask")
        # if self.mask is None:
        #      self.mask = tf.Variable(initial_value=mask, trainable=False)
        #  else:
        #      self.mask.assign(mask)
        self.mask.assign(mask)

    def __call__(self, w):
        """
        Used by tensorflow to add the constraint to the computation graph.
        """
        logger.debug("__call__")
        return tf.multiply(w, self.mask)