########################################################################################
########################################################################################
#--------------------------------------------------------------------------------------#
# This file contains preliminary testing of the BUTTER-CNN code.                       #
# Created by Erik Bensen                                                               #    
# Updated 2022-10-14                                                                   #
#--------------------------------------------------------------------------------------#
########################################################################################
########################################################################################

import tensorflow as tf
from cnn.cnn_net import make_net 

# set seed
tf.random.set_seed(0)

# test parameters
cell_info_dict = {
        'conv3x3':{'type':'paralleladd', 'nodes':1, 'operations':[['conv3x3']]},
        'conv5x5':{'type':'paralleladd', 'nodes':1, 'operations':[['conv5x5']]},
        'sepconv3x3':{'type':'paralleladd', 'nodes':1, 'operations':[['sepconv3x3']]},
        'sepconv5x5':{'type':'paralleladd', 'nodes':1, 'operations':[['sepconv5x5']]},
        'conv1x1':{'type':'paralleladd', 'nodes':1, 'operations':[['conv1x1']]},
        'resnet':{'type':'graph', 'nodes':3, 'operations':[['identity', 'conv3x3'],['identity']]},
        'inception':{'type':'parallelconcat', 'nodes':4, 'operations':[['conv5x5','conv1x1'],['conv3x3','conv1x1'],['conv1x1'],['conv1x1','maxpool3x3']]}
}
cell_info = cell_info_dict['inception']

downsamples = 1 
cell_depth = 2 
channels = [16, 32]
dataset = 'cifar10'
batch_norm = True

# Load cifar10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create model
model = make_net(cell_info, downsamples, cell_depth, channels, dataset, batch_norm)

# Compile model
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# Fit model
model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))

# Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])