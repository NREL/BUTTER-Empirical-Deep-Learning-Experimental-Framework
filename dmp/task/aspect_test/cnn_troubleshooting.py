# File to troubleshoot the CNN functions in aspect test utils
from dmp.task.aspect_test.aspect_test_utils import make_conv_network, make_keras_network_from_network_module

options = {
    'input_shape': [28, 28, 3],
    'downsamples': 1,
    'widths': [16, 32],
    'input_activation': None,
    'internal_activation': 'relu',
    'output_activation': 'softmax',
    'cell_depth': 2,
    'cell_type': 'graph',
    'cell_nodes': 3,
    'cell_ops': [['conv3x3', 'maxpool3x3'], ['conv1x1']],
    'classes': 10,
    'batch_norm': False
}

net_module = make_conv_network(**options)
net = make_keras_network_from_network_module(net_module)
net.summary()
