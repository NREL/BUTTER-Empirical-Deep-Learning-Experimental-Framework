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

num_inputs = 20
inner_size = 20
num_outputs = 20
num_samples = 1000000
bins = math.ceil(num_samples / 1000)

inputs = []
outputs = []
for i in range(num_samples):
    # input = numpy.random.randn(numInputs)
    input = numpy.random.randn(num_inputs)
    weights = numpy.random.randn(num_inputs)
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


inputs_array = numpy.stack(inputs)
input_magnitudes = numpy.sqrt(numpy.sum(inputs_array ** 2, axis=1))

print('input range:', numpy.min(input_magnitudes), numpy.max(input_magnitudes))

outputs_array = numpy.stack(outputs)
output_magnitudes = numpy.sqrt(numpy.sum(outputs_array ** 2, axis=1))

print('output range:', numpy.min(output_magnitudes), numpy.max(output_magnitudes))

print('outputsArray')
pprint(outputs_array.flatten().shape)
print('outputsArray')

pyplot.clf()

pyplot.hist(input_magnitudes, bins)
pyplot.title('inputMagnitudes')
pyplot.show()
pyplot.hist(output_magnitudes, bins)
pyplot.title('outputMagnitudes')
pyplot.show()

pyplot.hist(inputs_array.flatten(), bins)
pyplot.title('inputs {} {}'.format(numpy.mean(inputs_array.flatten()), numpy.var(inputs_array.flatten())))
pyplot.show()

pyplot.hist(outputs_array.flatten(), bins)


pyplot.title('outputs {} {}'.format(numpy.mean(outputs_array.flatten()), numpy.var(outputs_array.flatten())))
pyplot.show()

pyplot.plot(input_magnitudes, output_magnitudes, 'o', color='black')
pyplot.title('magnitude transfer')
pyplot.show()

pyplot.plot(inputs_array[:, 0], outputs_array[:, 0], 'o', color='black')
pyplot.title('variable transfer')
pyplot.show()
pyplot.show()
