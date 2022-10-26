# File to troubleshoot the CNN functions in aspect test utils
from aspect_test_utils import make_conv_network, make_keras_network_from_network_module
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import activations

options = {
    'input_channels': 3,
    'downsamples': 1,
    'widths': [16, 32],
    'input_activation': None,
    'internal_activation': 'relu',
    'output_activation': 'softmax',
    'cell_depth': 2,
    'cell_type': 'paralleladd',
    'cell_nodes': 1,
    'cell_ops': [['conv3x3']],
    'classes': 10,
    'batch_norm': False
}

net_module = make_conv_network(**options)
print(net_module)
