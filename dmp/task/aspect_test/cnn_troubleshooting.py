# File to troubleshoot the CNN functions in aspect test utils
# from dmp.layer.visitor.make_keras_network_from_module import make_keras_network_from_network_module
# from dmp.task.aspect_test.aspect_test_utils import make_conv_network

# import tensorflow.keras as keras

import simplejson
from dmp import jobqueue_interface
from dmp.layer import *

options = {
    'input_shape': [28, 28, 3],
    'downsamples': 1,
    'widths': [16, 32],
    'input_activation': None,
    'internal_activation': 'relu',
    'output_activation': 'softmax',
    'cell_depth': 2,
    'cell_type': 'graph',  # 'paralleladd', 'parallelconcat', 'graph'
    'cell_nodes': 3,
    'cell_ops': [['conv1x1'], ['conv3x3', 'maxpool3x3']],
    'classes': 10,
    'batch_norm': 'none'
}
'''
A input
    -> B conv1x1

[A, sum(B)]
--
A input
    -> C conv3x3
    -> B conv1x1 
        -> D maxpool3x3

[A, sum(B), sum(C, D)]

sum(C, D) =
sum(conv3x3(input), maxpool3x3(conv1x1(input)))
'''

# net_module = make_conv_network(**options)
# inputs, outputs, node_layer_map = \
#     make_keras_network_from_network_module(net_module)
# net = keras.Model(inputs=inputs, outputs=outputs)
# net.summary()

#{"class": "Add", "inputs": [{"class": "DenseConv", "kernel_size": [3, 3], "strides": [1, 1]}, {"class": "MaxPool", "inputs": [{"class": "DenseConv", "filters": -1, "kernel_size": [1, 1], "strides": [1, 1]}], "pool_size": [3, 3], "strides": [1, 1]}]}
#{"class": "Add", "inputs": [{"class": "DenseConv", "kernel_size": [3, 3]}, {"class": "MaxPool", "inputs": [{"class": "DenseConv", "kernel_size": [1, 1]}], "pool_size": [3, 3]}]}

l = Add({}, [
    DenseConv.make(-1, [3, 3], [1, 1], {}, []),
    MaxPool.make([3, 3], [1, 1], {},
                 [DenseConv.make(-1, [1, 1], [1, 1], {}, [])])
])

l = Add({}, [
    DenseConv.make3x3(),
    MaxPool.make3x3(DenseConv.make1x1())
])

m = jobqueue_interface.jobqueue_marshal.marshal(l)
j = simplejson.dumps(m, sort_keys=True)
print(j)



