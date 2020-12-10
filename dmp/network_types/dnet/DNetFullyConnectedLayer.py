import numpy

from dmp.network_types.dnet.DNetNode import DNetNode


class DNetFullyConnectedLayer:
    
    def __init__(self, numInputs: int, numPoints: int, numOutputs: int, transferFunction):
        print('DNetFullyConnectedLayer({}*{} -> {})'.format(numInputs, numPoints, numOutputs))
        self.nodes = [DNetNode(numInputs, numPoints, 1, transferFunction) for i in range(numOutputs)]
    
    def compute(self, input: numpy.ndarray):
        nodes = self.nodes
        numNodes = len(nodes)
        output = numpy.empty(numNodes)
        for i in range(numNodes):
            output[i] = nodes[i].compute(input)
        return output
    
    @property
    def numParameters(self):
        return sum((node.numParameters for node in self.nodes))
    
    def getFlatParameters(self, parameters):
        return self.pageThroughNodes(lambda node, offset: node.getFlatParameters(parameters[offset:]))
    
    def setFlatParameters(self, parameters):
        return self.pageThroughNodes(lambda node, offset: node.setFlatParameters(parameters[offset:]))
    
    def pageThroughNodes(self, function):
        offset = 0
        for node in self.nodes:
            offset += function(node, offset)
        return offset
