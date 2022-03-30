"""
Tensorflow Datasets Homepage: https://www.tensorflow.org/datasets
Catalog: https://www.tensorflow.org/datasets/catalog/mnist
Source: https://github.com/tensorflow/datasets

Name, Data Type, Task, Feature Types, # Observations, # Features



Example from: https://www.tensorflow.org/datasets/keras_example
"""

# !pip install tensorflow-datasets

import numpy
import tensorflow_datasets

# Load MNIST
(ds_train, ds_test), ds_info = tensorflow_datasets.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True)

data = list(tensorflow_datasets.as_numpy(ds_train))
# print(len(data))
# list of tuples of arrays?
# pprint(data[0])

input_list = []
output_list = []
for input, output in data:
    input_list.append(input)
    output_list.append(output)

outputs = numpy.stack(output_list)
inputs = numpy.array(input_list)
print(inputs.shape)
print(outputs.shape)
