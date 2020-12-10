import numpy

from dmp.network_types.dnet.dnet_fully_connected_layer import DNetFullyConnectedLayer


class DNetFullyConnectedNetwork:
    
    def __init__(self, layer_configs: [(int, int)], transfer_function):
        self.layers: [DNetFullyConnectedLayer] = []
        num_layers = len(layer_configs) - 1
        if num_layers < 1:
            raise Exception('Invalid layer configuration.')
        
        print('DNetFullyConnectedNetwork {} '.format(layer_configs))
        for i in range(num_layers):
            num_inputs = layer_configs[i][0]
            num_points = layer_configs[i][1]
            num_outputs = layer_configs[i + 1][0]
            print('make layer {}: {}*{} -> {} '.format(i, num_inputs, num_points, num_outputs))
            layer = DNetFullyConnectedLayer(num_inputs, num_points, num_outputs, transfer_function)
            self.layers.append(layer)
    
    def initialize(self):
        # randomly initialize layers
        for layer in self.layers:
            layer.positions = numpy.random.rand(*layer.positions.shape)
    
    def compute(self, input: numpy.ndarray):
        value = input
        for layer in self.layers:
            value = layer.compute(value)
        return value
    
    @property
    def num_parameters(self):
        return sum((layer.num_parameters for layer in self.layers))
    
    def set_flat_parameters(self, parameters):
        self.page_through_layers(lambda layer, offset: layer.set_flat_parameters(parameters[offset:]))
    
    def get_flat_parameters(self, parameters):
        self.page_through_layers(lambda layer, offset: layer.get_flat_parameters(parameters[offset:]))
    
    def page_through_layers(self, function):
        offset = 0
        for layer in self.layers:
            offset += function(layer, offset)
        return offset
