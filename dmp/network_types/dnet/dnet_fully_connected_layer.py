import numpy

from dmp.network_types.dnet.dnet_node import DNetNode


class DNetFullyConnectedLayer:
    
    def __init__(self, num_inputs: int, num_points: int, num_outputs: int, transfer_function):
        print('DNetFullyConnectedLayer({}*{} -> {})'.format(num_inputs, num_points, num_outputs))
        self.nodes = [DNetNode(num_inputs, num_points, 1, transfer_function) for i in range(num_outputs)]
    
    def compute(self, input: numpy.ndarray):
        nodes = self.nodes
        num_nodes = len(nodes)
        output = numpy.empty(num_nodes)
        for i in range(num_nodes):
            output[i] = nodes[i].compute(input)
        return output
    
    @property
    def num_parameters(self):
        return sum((node.num_parameters for node in self.nodes))
    
    def get_flat_parameters(self, parameters):
        return self.page_through_nodes(lambda node, offset: node.get_flat_parameters(parameters[offset:]))
    
    def set_flat_parameters(self, parameters):
        return self.page_through_nodes(lambda node, offset: node.set_flat_parameters(parameters[offset:]))
    
    def page_through_nodes(self, function):
        offset = 0
        for node in self.nodes:
            offset += function(node, offset)
        return offset
