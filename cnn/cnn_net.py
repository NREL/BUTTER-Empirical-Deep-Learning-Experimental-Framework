########################################################################################
########################################################################################
#--------------------------------------------------------------------------------------#
# This file contains the functions for creating nets in BUTTER-CNN.                    #
# Created by Erik Bensen                                                               #    
# Updated 2022-10-20                                                                   #
#--------------------------------------------------------------------------------------#
########################################################################################
########################################################################################

import tensorflow as tf
import tensorflow.keras.layers as layers 
import tensorflow.keras.backend as K
import tensorflow.keras.models as models
import tensorflow.keras.regularizers as regularizers
from cnn.cell_structures import make_graph_cell, make_parallel_concat_cell, make_parallel_add_cell, DownsampleCell, FinalClassifier, ConvStem

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Get cell function                                             #
#--------------------------------------------------------------------------------------#
########################################################################################

def get_cell(type='graph', nodes=2, operations=None, channels=16, batch_norm=False, activation='relu', **kwargs):
    if type == 'graph':
        return make_graph_cell(nodes, operations, channels, batch_norm, activation, **kwargs)
    elif type == 'parallelconcat':
        return make_parallel_concat_cell(nodes, operations, channels, batch_norm, activation, **kwargs)
    elif type == 'paralleladd':
        return make_parallel_add_cell(nodes, operations, channels, batch_norm, activation, **kwargs)
    else:
        raise ValueError(f'Cell type {type} not recognized')

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

def make_net(cell_info={'type':'graph', 'nodes':2, 'operations':[['conv3x3']]}, downsamples=1, cell_depth=2, channels=[16,16], dataset='cifar10', 
            batch_norm=False, activation='relu', out_activation='softmax'):
    # Check downsamples and cell depth and channels
    if downsamples > max_downsamples_dict[dataset]:
        raise ValueError(f'Cannot have more downsamples than {max_downsamples_dict[dataset]} for {dataset} dataset')
    if cell_depth < downsamples + 1:
        raise ValueError(f'Cell depth must be at least {downsamples + 1} for {downsamples} downsamples')
    if len(channels) != downsamples + 1:
        raise ValueError(f'Number of channels must have {downsamples + 1} elements for {downsamples} downsamples')

    # Get input size and number of classes
    input_size = input_size_dict[dataset]
    classes = classes_dict[dataset]

    # Get num cells per stack 
    stack_cells = [cell_depth // (downsamples + 1) for _ in range(downsamples + 1)]
    for i in range(cell_depth % (downsamples + 1)):
        stack_cells[-i-1] += 1

    # Create sequential net
    net = models.Sequential()
    net.add(layers.InputLayer(input_size))
    net.add(ConvStem(channels[0], batch_norm, activation))
    for i in range(downsamples+1):
        for j in range(stack_cells[i]):
            net.add(get_cell(**cell_info, channels=channels[i], batch_norm=batch_norm, activation=activation))
        if i < downsamples:
            net.add(DownsampleCell(channels[i+1], activation=activation))
    net.add(FinalClassifier(classes, activation=out_activation))
    return net


########################################################################################
#--------------------------------------------------------------------------------------#
#                        Informative functions                                         #
#--------------------------------------------------------------------------------------#
########################################################################################

def get_num_trainable_params(net):
    return net.count_params()