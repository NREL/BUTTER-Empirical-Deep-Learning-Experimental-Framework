'''


+ input random samples (uniform? gaussian?)
+ inspect output distribution
    + dimensional graphs
    + aggregate sensitivity
        + wiggle input and see if output magnitude is similar
            + graph input delta vs output delta



'''

import numpy
from matplotlib import pyplot

from dmp.network_types import DNetFullyConnectedNetwork
from dmp.network_types.dnet.dnet_node import DNetNode

num_inputs = 20
inner_size = 20
num_outputs = 20
num_samples = 5000

num_points = 1
# transferFunction = DNetNode.idw
# transferFunction = DNetNode.inverseSquaredDistance
# transferFunction = DNetNode.nearestNeighbour
transfer_function = DNetNode.scratch
# transferFunction = DNetNode.unnormalizedInverseSquaredDistance
# transferFunction = DNetNode.inverseQuadraticDistanceWeighting
# transferFunction = lambda node, input: DNetNode.rbfInverseQuadratic(node, input)
network = DNetFullyConnectedNetwork(
    [
        (num_inputs, num_points),
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
        (num_outputs, 0)
        ],
    transfer_function)

# initialize
# network.setFlatParameters(numpy.random.randn(network.numParameters))
network.set_flat_parameters(numpy.random.rand(network.num_parameters))
# network.setFlatParameters(numpy.ones(network.numParameters))

inputs = []
outputs = []
for i in range(num_samples):
    # input = numpy.random.randn(numInputs)
    input = numpy.random.rand(num_inputs)
    # input /= numpy.sum(input**2)
    output = network.compute(input)
    # print('input')
    # pprint(input)
    # print('output')
    # pprint(output)
    inputs.append(input)
    outputs.append(output)

inputs_array = numpy.stack(inputs)
input_magnitudes = numpy.sqrt(numpy.sum(inputs_array ** 2, axis=1))

print('input range:', numpy.min(input_magnitudes), numpy.max(input_magnitudes))

outputs_array = numpy.stack(outputs)
output_magnitudes = numpy.sqrt(numpy.sum(outputs_array ** 2, axis=1))

print('output range:', numpy.min(output_magnitudes), numpy.max(output_magnitudes))
pyplot.clf()

pyplot.hist(input_magnitudes)
pyplot.title('inputMagnitudes')
pyplot.show()
pyplot.hist(output_magnitudes)
pyplot.title('outputMagnitudes')
pyplot.show()

pyplot.hist(inputs_array)
pyplot.title('inputs {} {}'.format(numpy.mean(inputs_array.flatten()), numpy.var(inputs_array.flatten())))
pyplot.show()

pyplot.hist(outputs_array)
pyplot.title('outputs {} {}'.format(numpy.mean(outputs_array.flatten()), numpy.var(outputs_array.flatten())))
pyplot.show()

pyplot.plot(input_magnitudes, output_magnitudes, 'o', color='black')
pyplot.title('magnitude transfer')
pyplot.show()

pyplot.plot(inputs_array[:, 0], outputs_array[:, 0], 'o', color='black')
pyplot.title('variable transfer')
pyplot.show()
pyplot.show()
