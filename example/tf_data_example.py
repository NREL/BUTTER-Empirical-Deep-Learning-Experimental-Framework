"""
Tensorflow Datasets Homepage: https://www.tensorflow.org/datasets
Catalog: https://www.tensorflow.org/datasets/catalog/mnist
Source: https://github.com/tensorflow/datasets

Name, Data Type, Task, Feature Types, # Observations, # Features



Example from: https://www.tensorflow.org/datasets/keras_example
"""

# !pip install tensorflow-datasets
import tensorflow_datasets
import tensorflow

# Load MNIST
(ds_train, ds_test), ds_info = tensorflow_datasets.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True)

fig = tensorflow_datasets.show_examples(ds_train, ds_info)

# # Build your input pipeline
# ds = ds.shuffle(1024).batch(32).prefetch(tensorflow.data.experimental.AUTOTUNE)
# for example in ds.take(1):
#   image, label = example["image"], example["label"]

print(ds_info)
print(ds_info.features["label"].num_classes)
print(ds_info.features["label"].names)
print(ds_info.features["label"].int2str(7))  # Human readable version (8 -> 'cat')
print(ds_info.features["label"].str2int('7'))

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.relu, input_shape=(4,)),  # input shape required
    tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.relu),
    tensorflow.keras.layers.Dense(3)
    ])


# Build training pipeline --------------------------------------------

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tensorflow.cast(image, tensorflow.float32) / 255., label

# TFDS provide the images as tensorflow.uint8, while the model expect tensorflow.float32, so normalize images
ds_train = ds_train.map(
    normalize_img,
    num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)

# As the dataset fit in memory, cache before shuffling for better performance.
ds_train = ds_train.cache()

# For true randomness, set the shuffle buffer to the full dataset size.
# Note: For bigger datasets which do not fit in memory, a standard value
# is 1000 if your system allows it
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

# Batch after shuffling to get unique batches at each epoch.
ds_train = ds_train.batch(128)

# Good practice to end the pipeline by prefetching for performances.
ds_train = ds_train.prefetch(tensorflow.data.experimental.AUTOTUNE)

# Build evaluation pipeline --------------------------------------------
ds_test = ds_test.map(
    normalize_img,
    num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tensorflow.data.experimental.AUTOTUNE)

# Create and train the model --------------------------------------------

model = tensorflow.keras.models.Sequential([
  tensorflow.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tensorflow.keras.layers.Dense(128,activation='relu'),
  tensorflow.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tensorflow.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    ds_train,
    epochs=12,
    validation_data=ds_test,
)