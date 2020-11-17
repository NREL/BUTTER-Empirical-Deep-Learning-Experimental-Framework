'''


+ input random samples (uniform? gaussian?)
+ inspect output distribution
    + dimensional graphs
    + aggregate sensitivity
        + wiggle input and see if output magnitude is similar
            + graph input delta vs output delta



'''
import math
from pprint import pprint

import numpy
from matplotlib import pyplot

from network_types.dnet.DNetFullyConnectedNetwork import DNetFullyConnectedNetwork
from network_types.dnet.DNetNode import DNetNode

numInputs = 20
innerSize = 20
numOutputs = 20
numSamples = 1000000
bins = math.ceil(numSamples / 1000)

inputs = []
outputs = []
for i in range(numSamples):
    # input = numpy.random.randn(numInputs)
    input = numpy.random.randn(numInputs)
    weights = numpy.random.randn(numInputs)
    weights /= numpy.linalg.norm(weights)
    projection = numpy.dot(input, weights)
    
    # position = numpy.random.randn(1)
    
    # position = numpy.random.randn(numInputs)
    # delta = projection - position
    # d = 1.0 / numpy.abs(delta)
    d = projection
    
    # d = numpy.amax(delta)
    # d = numpy.sum(numpy.abs(delta))
    
    
    output = d.reshape((d.size, 1))
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

print('outputsArray')
pprint(outputsArray.flatten().shape)
print('outputsArray')

pyplot.clf()

pyplot.hist(inputMagnitudes, bins)
pyplot.title('inputMagnitudes')
pyplot.show()
pyplot.hist(outputMagnitudes, bins)
pyplot.title('outputMagnitudes')
pyplot.show()

pyplot.hist(inputsArray.flatten(), bins)
pyplot.title('inputs {} {}'.format(numpy.mean(inputsArray.flatten()), numpy.var(inputsArray.flatten())))
pyplot.show()

pyplot.hist(outputsArray.flatten(), bins)


pyplot.title('outputs {} {}'.format(numpy.mean(outputsArray.flatten()), numpy.var(outputsArray.flatten())))
pyplot.show()

pyplot.plot(inputMagnitudes, outputMagnitudes, 'o', color='black')
pyplot.title('magnitude transfer')
pyplot.show()

pyplot.plot(inputsArray[:, 0], outputsArray[:, 0], 'o', color='black')
pyplot.title('variable transfer')
pyplot.show()
pyplot.show()
