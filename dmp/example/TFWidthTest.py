"""
Tensorflow Datasets Homepage: https://www.tensorflow.org/datasets
Catalog: https://www.tensorflow.org/datasets/catalog/mnist
Source: https://github.com/tensorflow/datasets

Name, Data Type, Task, Feature Types, # Observations, # Features



Example from: https://www.tensorflow.org/datasets/keras_example
"""
from pprint import pprint

import numpy
import tensorflow
import tensorflow_datasets
from matplotlib import pyplot
from tensorflow import keras

depths = range(1, 5)
width = 64
histories = []
for depth in depths:
    # load dataset --------------------------------------------
    # (trainingDataset, validationDataset), ds_info = tensorflow_datasets.load(
    #     # 'mnist',
    #     'wine_quality',
    #     split=['train', 'test'],
    #     as_supervised=True,
    #     shuffle_files=True,
    #     with_info=True)
    
    dataset, ds_info = tensorflow_datasets.load(
        # 'mnist',
        'wine_quality',
        split='train',
        as_supervised=True,
        shuffle_files=True,
        with_info=True)
    size = ds_info.splits['train'].num_examples
    
    
    trainingSize = int(0.8 * size)
    validationSize = int(0.2 * size)
    # test_size = int(0.1 * size)

    print('size: {}, train size: {}, validation size: {}'.format(size, trainingSize, validationSize))
    
    dataset = dataset.shuffle(size)
    trainingDataset = dataset.take(trainingSize)
    validationDataset = dataset.skip(trainingSize)
    
    numFeatures = 0
    numOutputs = 0
    for element in dataset:
        print(element)
        numFeatures = len(element[0])
        outputs = element[1]
        if isinstance(outputs, tensorflow.Tensor):
            numOutputs = 1
        else:
            numOutputs = len(outputs)
        break
        
    print('num features: {}, num outputs: {}', numFeatures, numOutputs)
    
    # Build training pipeline --------------------------------------------
    
    # def normalize_img(image, label):
    #     """Normalizes images: `uint8` -> `float32`."""
    #     return tensorflow.cast(image, tensorflow.float32) / 255., label
    
    # # TFDS provide the images as tensorflow.uint8, while the model expect tensorflow.float32, so normalize images
    # trainingDataset = trainingDataset.map(
    #     normalize_img,
    #     num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
    
    # trainingDataset = trainingDataset.map(
    #         normalize_img,
    #         num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
    
    # As the dataset fit in memory, cache before shuffling for better performance.
    trainingDataset = trainingDataset.cache()
    
    # Batch after shuffling to get unique batches at each epoch.
    trainingDataset = trainingDataset.batch(128)
    
    # Good practice to end the pipeline by prefetching for performances.
    trainingDataset = trainingDataset.prefetch(tensorflow.data.experimental.AUTOTUNE)
    
    # Build evaluation pipeline --------------------------------------------
    # validationDataset = validationDataset.map(
    #     normalize_img,
    #     num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
    validationDataset = validationDataset.batch(128)
    validationDataset = validationDataset.cache()
    validationDataset = validationDataset.prefetch(tensorflow.data.experimental.AUTOTUNE)
    
    # Create and train the model --------------------------------------------
    
    inputs = [keras.Input(shape=(1,)) for i in range(numFeatures)]
    concatenatedInputs = keras.layers.Concatenate()(inputs)
    # layers = []
    # layers.append(tensorflow.keras.layers.Flatten(input_shape=(numFeatures,)))
    # layers.append(tensorflow.keras.layers.Flatten(input_shape=(28, 28, 1)))
    # layers.append(tensorflow.keras.layers.Concatenate())
    # layers.append(keras.layers.Dense(width, activation='relu', input_dim=numFeatures))
    x = concatenatedInputs
    for i in range(depth - 1):
        x = keras.layers.Dense(width, activation='relu')(x)
    # layers.append(tensorflow.keras.layers.Dense(10, activation='softmax'))
    # outputs = tensorflow.keras.layers.Dense(numOutputs, activation='softmax')
    output = keras.layers.Dense(1, activation='relu')(x)
    output = keras.layers.experimental.preprocessing.Rescaling(scale=10.0, offset=5)(output)
    #
    # outputs = keras.models.Sequential(layers)
    model = keras.Model(inputs=inputs, outputs=output)
    # model.compile(
    #     loss='sparse_categorical_crossentropy',
    #     optimizer=keras.optimizers.Adam(0.001),
    #     metrics=['accuracy'],
    #     )

    # keras.utils.plot_model(model, show_shapes=True)
    
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
        )
    
    trainingHistory = model.fit(
        trainingDataset,
        epochs=20,
        validation_data=validationDataset,
        )
    
    histories.append(trainingHistory)

for history in histories:
    print(history.history)


# # val_loss, loss, accuracy, val_accuracy
def extractVariableFromHistories(name: str, histories: [tensorflow.keras.callbacks.History]) -> numpy.ndarray:
    return numpy.stack([history.history[name] for history in histories])


validationAccuracies = extractVariableFromHistories('val_accuracy', histories)

# validationAccuracies = numpy.array([[0.9023, 0.91289997, 0.91670001, 0.92089999, 0.92290002,
#                                      0.92320001, 0.92519999, 0.92439997, 0.92729998, 0.9267,
#                                      0.92680001, 0.9271, 0.9271, 0.92900002, 0.92769998,
#                                      0.92799997, 0.9267, 0.92729998, 0.92799997, 0.92750001,
#                                      0.92629999, 0.9278, 0.92659998, 0.92900002, 0.92720002],
#                                     [0.94590002, 0.96100003, 0.96820003, 0.97320002, 0.97350001,
#                                      0.9774, 0.97680002, 0.9774, 0.977, 0.9774,
#                                      0.97750002, 0.9781, 0.97960001, 0.97680002, 0.97729999,
#                                      0.97930002, 0.97920001, 0.97899997, 0.97820002, 0.97890002,
#                                      0.97839999, 0.97680002, 0.97839999, 0.98019999, 0.97909999],
#                                     [0.95539999, 0.96619999, 0.96939999, 0.97640002, 0.9774,
#                                      0.97539997, 0.97549999, 0.9788, 0.97920001, 0.97759998,
#                                      0.97469997, 0.97549999, 0.97979999, 0.97640002, 0.97930002,
#                                      0.9799, 0.97939998, 0.97680002, 0.9799, 0.9795,
#                                      0.9799, 0.97369999, 0.97899997, 0.97920001, 0.97780001],
#                                     [0.9587, 0.96630001, 0.96829998, 0.972, 0.97479999,
#                                      0.97710001, 0.977, 0.97719997, 0.97600001, 0.97320002,
#                                      0.97229999, 0.97460002, 0.97600001, 0.97619998, 0.97970003,
#                                      0.97869998, 0.97780001, 0.97710001, 0.97920001, 0.97659999,
#                                      0.97909999, 0.977, 0.97659999, 0.98000002, 0.97710001],
#                                     [0.96060002, 0.97000003, 0.96160001, 0.97359997, 0.97780001,
#                                      0.97799999, 0.9716, 0.9781, 0.97589999, 0.97689998,
#                                      0.97909999, 0.97850001, 0.977, 0.97390002, 0.98100001,
#                                      0.97589999, 0.97850001, 0.97670001, 0.97909999, 0.97930002,
#                                      0.98079997, 0.97820002, 0.977, 0.98199999, 0.97619998],
#                                     [0.95279998, 0.96969998, 0.97149998, 0.97140002, 0.97460002,
#                                      0.97570002, 0.97549999, 0.97390002, 0.97790003, 0.97909999,
#                                      0.97729999, 0.97539997, 0.97289997, 0.97759998, 0.97659999,
#                                      0.97970003, 0.97729999, 0.97490001, 0.97979999, 0.97689998,
#                                      0.97610003, 0.97750002, 0.97890002, 0.97850001, 0.9788],
#                                     [0.95340002, 0.96649998, 0.97219998, 0.97369999, 0.97229999,
#                                      0.9716, 0.97600001, 0.97180003, 0.97250003, 0.97610003,
#                                      0.9763, 0.9763, 0.97890002, 0.9774, 0.97689998,
#                                      0.9795, 0.97799999, 0.97850001, 0.9777, 0.97670001,
#                                      0.97909999, 0.977, 0.97579998, 0.97839999, 0.97799999],
#                                     [0.9515, 0.96810001, 0.96569997, 0.9745, 0.97299999,
#                                      0.9756, 0.97589999, 0.97600001, 0.97839999, 0.97350001,
#                                      0.97640002, 0.97359997, 0.97600001, 0.98049998, 0.97310001,
#                                      0.9777, 0.97570002, 0.97780001, 0.97820002, 0.97930002,
#                                      0.9774, 0.9795, 0.97960001, 0.97799999, 0.97829998],
#                                     [0.94840002, 0.963, 0.9698, 0.97289997, 0.97189999,
#                                      0.9763, 0.97729999, 0.9666, 0.9745, 0.97420001,
#                                      0.97790003, 0.97670001, 0.98000002, 0.97750002, 0.97850001,
#                                      0.97729999, 0.97680002, 0.97780001, 0.98009998, 0.97500002,
#                                      0.97829998, 0.98199999, 0.98199999, 0.97909999, 0.97579998],
#                                     [0.94819999, 0.96469998, 0.96640003, 0.97289997, 0.97460002,
#                                      0.97390002, 0.972, 0.97579998, 0.97899997, 0.9745,
#                                      0.97689998, 0.977, 0.97750002, 0.977, 0.97299999,
#                                      0.96820003, 0.9777, 0.97539997, 0.97829998, 0.97829998,
#                                      0.97640002, 0.97680002, 0.97799999, 0.97729999, 0.9774]])

pprint(validationAccuracies)

minimumValidationAccuracyIndices = numpy.argmax(validationAccuracies, axis=1)
pprint(minimumValidationAccuracyIndices)
minimumValidationAccuracies = numpy.max(validationAccuracies, axis=1)
pprint(minimumValidationAccuracies)
validationErrors = 1 - minimumValidationAccuracies

# minValidationLossIndices = [numpy.argmin(history.history['val_loss']) for history in histories]
# minValidationLoss = [history.history['val_loss'][i] for i, history in enumerate(histories)]
#
# minValidationLossIndices = [numpy.argmin(history.history['val_loss']) for history in histories]
#
# for history in histories:
#     # print(history)
#     # validation_loss_index = numpy.argmin(history.history['loss'])
#     minValidationLossIndex = numpy.argmin(history.history['val_loss'])
#     minValidationLossIndicies.append(minValidationLossIndex)
#     minValidationLoss = history.history['val_loss'][minValidationLossIndex]
#     minValidationLossIndicies.append(minValidationLoss)
#     #['val_acc']  ['acc']
#
#
pyplot.style.use('seaborn-whitegrid')

figure, ax = pyplot.subplots()
ax.plot(depths, minimumValidationAccuracyIndices)
ax.set_xlabel('depth')
ax.set_ylabel('epoch of minimum validation error')
pyplot.show()

figure, ax = pyplot.subplots()
ax.plot(depths, validationErrors)
ax.set_xlabel('depth')
ax.set_ylabel('minimum validation error')
pyplot.show()
