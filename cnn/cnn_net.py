########################################################################################
########################################################################################
#--------------------------------------------------------------------------------------#
# This file contains the functions for creating nets in BUTTER-CNN.                    #
# Created by Erik Bensen                                                               #
# Updated 2022-10-20                                                                   #
#--------------------------------------------------------------------------------------#
########################################################################################
########################################################################################

from typing import List, Any, Dict
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


def make_cell(
    type: str = 'graph',
    nodes: int = 2,
    operations=None,
    filters: int = 16,
    batch_norm: str = 'none',
    activation: str = 'relu',
    **kwargs,
):
    if type == 'graph':
        return make_graph_cell(
            nodes,
            operations,
            filters,
            batch_norm,
            activation,
            **kwargs,
        )
    elif type == 'parallelconcat':
        return make_parallel_concat_cell(
            nodes,
            operations,
            filters,
            batch_norm,
            activation,
            **kwargs,
        )
    elif type == 'paralleladd':
        return make_parallel_add_cell(
            nodes,
            operations,
            filters,
            batch_norm,
            activation,
            **kwargs,
        )
    else:
        raise ValueError(f'Cell type {type} not recognized')


########################################################################################
#--------------------------------------------------------------------------------------#
#                        Make net function                                             #
#--------------------------------------------------------------------------------------#
########################################################################################

input_size_dict = {
    'cifar10': [32, 32, 3],
    'cifar100': [32, 32, 3],
    'imagenet': [224, 224, 3],
    'mnist': [28, 28, 1],
}
classes_dict = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'mnist': 10,
}
max_downsamples_dict = {
    'cifar10': 5,
    'cifar100': 5,
    'imagenet': 7,
    'mnist': 4,
}


def make_net(
    cell_info: Dict[str, Any] = {
        'type': 'graph',
        'nodes': 2,
        'operations': [['conv3x3']]
    },
    downsamples: int = 1,
    cell_depth: int = 2,
    filters: List[int] = [16, 16],
    dataset: str = 'cifar10',
    batch_norm: str = 'none',
    activation: str = 'relu',
    output_activation: str = 'softmax',
):
    # Check downsamples and cell depth and filters
    if downsamples > max_downsamples_dict[dataset]:
        raise ValueError(
            f'Cannot have more downsamples than {max_downsamples_dict[dataset]} for {dataset} dataset'
        )
    if cell_depth < downsamples + 1:
        raise ValueError(
            f'Cell depth must be at least {downsamples + 1} for {downsamples} downsamples'
        )
    if len(filters) != downsamples + 1:
        raise ValueError(
            f'Number of filters must have {downsamples + 1} elements for {downsamples} downsamples'
        )

    # Get input size and number of classes
    input_size = input_size_dict[dataset]
    classes = classes_dict[dataset]

    # Get num cells per stack
    stack_cells = [
        cell_depth // (downsamples + 1) for _ in range(downsamples + 1)
    ]
    for i in range(cell_depth % (downsamples + 1)):
        stack_cells[-i - 1] += 1

    # Create sequential net
    net = models.Sequential()
    net.add(layers.InputLayer(input_size))
    net.add(ConvStem(filters[0], batch_norm, activation))
    for i in range(downsamples + 1):
        for j in range(stack_cells[i]):
            net.add(
                make_cell(**cell_info,
                          filters=filters[i],
                          batch_norm=batch_norm,
                          activation=activation))
        if i < downsamples:
            net.add(DownsampleCell(filters[i + 1], activation=activation))
    net.add(FinalClassifier(classes, activation=output_activation))
    return net


########################################################################################
#--------------------------------------------------------------------------------------#
#                        Informative functions                                         #
#--------------------------------------------------------------------------------------#
########################################################################################


def get_num_trainable_params(net):
    return net.count_params()