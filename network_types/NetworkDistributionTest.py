'''


+ input random samples (uniform? gaussian?)
+ inspect output distribution
    + dimensional graphs
    + aggregate sensitivity
        + wiggle input and see if output magnitude is similar
            + graph input delta vs output delta



'''
import math

import numpy
from matplotlib import pyplot

from network_types.dnet.DNetFullyConnectedNetwork import DNetFullyConnectedNetwork
from network_types.dnet.DNetNode import DNetNode

numInputs = 20
innerSize = 20
numOutputs = 20
numSamples = 5000

numPoints = 1
# transferFunction = DNetNode.idw
# transferFunction = DNetNode.inverseSquaredDistance
# transferFunction = DNetNode.nearestNeighbour
transferFunction = DNetNode.scratch
# transferFunction = DNetNode.unnormalizedInverseSquaredDistance
# transferFunction = DNetNode.inverseQuadraticDistanceWeighting
# transferFunction = lambda node, input: DNetNode.rbfInverseQuadratic(node, input)
network = DNetFullyConnectedNetwork(
    [
        (numInputs, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        # (innerSize, numPoints),
        (numOutputs, 0)
        ],
    transferFunction)

# initialize
# network.setFlatParameters(numpy.random.randn(network.numParameters))
network.setFlatParameters(numpy.random.rand(network.numParameters))
# network.setFlatParameters(numpy.ones(network.numParameters))

inputs = []
outputs = []
for i in range(numSamples):
    # input = numpy.random.randn(numInputs)
    input = numpy.random.rand(numInputs)
    # input /= numpy.sum(input**2)
    output = network.compute(input)
    # print('input')
    # pprint(input)
    # print('output')
    # pprint(output)
    inputs.append(input)
    outputs.append(output)

inputsArray = numpy.stack(inputs)
inputMagnitudes = numpy.sqrt(numpy.sum(inputsArray ** 2, axis=1))

print('input range:', numpy.min(inputMagnitudes), numpy.max(inputMagnitudes))

outputsArray = numpy.stack(outputs)
outputMagnitudes = numpy.sqrt(numpy.sum(outputsArray ** 2, axis=1))

print('output range:', numpy.min(outputMagnitudes), numpy.max(outputMagnitudes))
pyplot.clf()

pyplot.hist(inputMagnitudes)
pyplot.title('inputMagnitudes')
pyplot.show()
pyplot.hist(outputMagnitudes)
pyplot.title('outputMagnitudes')
pyplot.show()

pyplot.hist(inputsArray)
pyplot.title('inputs {} {}'.format(numpy.mean(inputsArray.flatten()), numpy.var(inputsArray.flatten())))
pyplot.show()

pyplot.hist(outputsArray)
pyplot.title('outputs {} {}'.format(numpy.mean(outputsArray.flatten()), numpy.var(outputsArray.flatten())))
pyplot.show()

pyplot.plot(inputMagnitudes, outputMagnitudes, 'o', color='black')
pyplot.title('magnitude transfer')
pyplot.show()

pyplot.plot(inputsArray[:, 0], outputsArray[:, 0], 'o', color='black')
pyplot.title('variable transfer')
pyplot.show()
pyplot.show()
