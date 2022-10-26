########################################################################################
########################################################################################
#--------------------------------------------------------------------------------------#
# This file contains the functions for creating nets in BUTTER-CNN.                    #
# Created by Erik Bensen                                                               #    
# Updated 2022-10-13                                                                   #
#--------------------------------------------------------------------------------------#
########################################################################################
########################################################################################

import tensorflow as tf
import tensorflow.keras.layers as layers 
import tensorflow.keras.backend as K
import tensorflow.keras.models as models
import tensorflow.keras.regularizers as regularizers
from cell_structures import *

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Get cell function                                             #
#--------------------------------------------------------------------------------------#
########################################################################################

cell_dict = {'3x3': Conv3x3Cell,
             '5x5': Conv5x5Cell,
             'sep3x3': SepConv3x3Cell,
             'sep5x5': SepConv5x5Cell,
             '3x3pool': Conv3x3PoolCell,
             '5x5pool': Conv5x5PoolCell,
             'sep3x3pool': SepConv3x3PoolCell,
             'sep5x5pool': SepConv5x5PoolCell,
             'inception': InceptionCell,
             'resnet': ResNetCell,
             'resnetxt': ResNetXTCell,
             'nasbench101': NASBench101Cell,
             'nasbench201': NASBench201Cell}

def get_cell(type='3x3', channels=16, batch_norm=False, activation='relu'):
    if type in cell_dict.keys():
        return cell_dict[type](channels, batch_norm, activation)
    else:
        raise ValueError('Cell type not recognized.')

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Make net function                                             #
#--------------------------------------------------------------------------------------#
########################################################################################

input_size_dict = {
    'cifar10':[32, 32, 3],
    'cifar100':[32, 32, 3],
    'imagenet':[224, 224, 3],
    'mnist':[28, 28, 1]
}
classes_dict = {    
    'cifar10':10,
    'cifar100':100,
    'imagenet':1000,
    'mnist':10
}
max_downsamples_dict = {
    'cifar10':5,
    'cifar100':5,
    'imagenet':7,
    'mnist':4
}

def make_net(cell_type='3x3', 
            downsamples=1, 
            cell_depth=2, 
            channels=16, 
            dataset='cifar10', 
            batch_norm=False, 
            activation='relu', 
            out_activation='softmax'):

    # Check compatability of inputs
    if dataset in input_size_dict.keys():
        input_size = input_size_dict[dataset]
        classes = classes_dict[dataset]
    else:
        raise ValueError('Dataset not recognized.')
    if len(channels) < downsamples+1:
        raise ValueError('Too few numbers of channels items specified.')
    elif len(channels) > downsamples+1:
        raise ValueError('Too many numbers of channels items specified.')
    if downsamples > max_downsamples_dict[dataset]:
        raise ValueError('Too many downsamples for specified dataset.')
    
    # Initialize inputs and conv stem
    net = models.Sequential()
    net.add(layers.InputLayer(input_size))
    net.add(ConvStem(channels[0], batch_norm, activation))

    # Determine cells per cell block
    cells = [cell_depth // (downsamples+1) for _ in range(downsamples + 1)]
    remainder = cell_depth % (downsamples+1)
    for i in range(remainder):
        cells[-(i+1)] += 1
    
    # Add cell blocks
    for block in range(len(cells)):
        for _ in range(cells[block]):
            net.add(get_cell(cell_type, channels[block], batch_norm, activation))
        if block < len(cells)-1:
            net.add(DownsampleCell(channels[block+1], activation))
    
    # Add final layers
    net.add(FinalClassifier(classes, out_activation))
    return net

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Informative functions                                         #
#--------------------------------------------------------------------------------------#
########################################################################################

def get_num_trainable_params(net):
    return net.count_params()