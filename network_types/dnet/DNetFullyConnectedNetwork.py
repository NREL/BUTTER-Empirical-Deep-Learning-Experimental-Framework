import numpy

from network_types.dnet.DNetFullyConnectedLayer import DNetFullyConnectedLayer


class DNetFullyConnectedNetwork:
    
    def __init__(self, layer_configs: [(int, int)], transferFunction):
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
            layer = DNetFullyConnectedLayer(num_inputs, num_points, num_outputs, transferFunction)
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
    def numParameters(self):
        return sum((layer.numParameters for layer in self.layers))
    
    def setFlatParameters(self, parameters):
        self.pageThroughLayers(lambda layer, offset: layer.setFlatParameters(parameters[offset:]))
    
    def getFlatParameters(self, parameters):
        self.pageThroughLayers(lambda layer, offset: layer.getFlatParameters(parameters[offset:]))
    
    def pageThroughLayers(self, function):
        offset = 0
        for layer in self.layers:
            offset += function(layer, offset)
        return offset
