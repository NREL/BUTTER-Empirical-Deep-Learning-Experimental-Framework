########################################################################################
########################################################################################
#--------------------------------------------------------------------------------------#
# This file contains the classes for different cell structures in BUTTER-CNN.          #
# Created by Erik Bensen                                                               #
# Updated 2022-10-20                                                                   #
#--------------------------------------------------------------------------------------#
########################################################################################
########################################################################################

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Activation Functions
#--------------------------------------------------------------------------------------#
########################################################################################

activation_dict = {'relu': keras.activations.relu,
                    'relu6': tf.nn.relu6,
                    'leaky_relu': keras.layers.LeakyReLU(),
                    'elu': keras.activations.elu,
                    'selu': keras.activations.selu,
                    'sigmoid': keras.activations.sigmoid,
                    'hard_sigmoid': keras.activations.hard_sigmoid,
                    'swish': keras.activations.swish,
                    'tanh': keras.activations.tanh,
                    'softplus': keras.activations.softplus,
                    'softsign': keras.activations.softsign,
                    'softmax': keras.activations.softmax,
                    }

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Operation Layers
#--------------------------------------------------------------------------------------#
########################################################################################

# 3x3 conv bn activation 
class Conv3x3Operation(layers.Layer):
    def __init__(self, channels=128, batch_norm=False, activation='relu', **kwargs):
        super(Conv3x3Operation, self).__init__()
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(channels, 3, padding='same', activation='linear', **kwargs)
        self.bn = layers.BatchNormalization() if batch_norm else None
        self.act = activation_dict[activation]
    
    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

# 5x5 conv bn activation
class Conv5x5Operation(layers.Layer):
    def __init__(self, channels=128, batch_norm=False, activation='relu', **kwargs):
        super(Conv5x5Operation, self).__init__()
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(channels, 5, padding='same', activation='linear', **kwargs)
        self.bn = layers.BatchNormalization() if batch_norm else None
        self.act = activation_dict[activation]
    
    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

# 3x3 separable conv bn activation
class SepConv3x3Operation(layers.Layer):
    def __init__(self, channels=128, batch_norm=False, activation='relu', **kwargs):
        super(SepConv3x3Operation, self).__init__()
        self.batch_norm = batch_norm
        self.conv = layers.SeparableConv2D(channels, 3, padding='same', activation='linear', **kwargs)
        self.bn = layers.BatchNormalization() if batch_norm else None
        self.act = activation_dict[activation]
    
    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

# 5x5 separable conv bn activation
class SepConv5x5Operation(layers.Layer):
    def __init__(self, channels=128, batch_norm=False, activation='relu', **kwargs):
        super(SepConv5x5Operation, self).__init__()
        self.batch_norm = batch_norm
        self.conv = layers.SeparableConv2D(channels, 5, padding='same', activation='linear')
        self.bn = layers.BatchNormalization() if batch_norm else None
        self.act = activation_dict[activation]
    
    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

# 1x1 conv bn activation
class Conv1x1Operation(layers.Layer):
    def __init__(self, channels=128, batch_norm=False, activation='relu', **kwargs):
        super(Conv1x1Operation, self).__init__()
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(channels, 1, padding='same', activation='linear')
        self.bn = layers.BatchNormalization() if batch_norm else None
        self.act = activation_dict[activation]
    
    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

# projection operation 
class ProjectionOperation(layers.Layer):
    def __init__(self, channels=128, batch_norm=False, activation='relu', **kwargs):
        super(ProjectionOperation, self).__init__()
        # batch norm and activation are not used in projection operation
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(channels, 1, padding='same', activation='linear', **kwargs)
    
    def call(self, x):
        x = self.conv(x)
        return x

# 3x3 max pooling
class MaxPool3x3Operation(layers.Layer):
    def __init__(self, channels=None, batch_norm=False, activation='relu'):
        # channels is not used here, but is included for consistency with other operations
        # batch_norm and activation are not used here, but are included for consistency with other operations
        super(MaxPool3x3Operation, self).__init__()
        self.pool = layers.MaxPool2D(3, strides=1, padding='same')
    
    def call(self, x):
        x = self.pool(x)
        return x

# 3x3 avg pooling
class AvgPool3x3Operation(layers.Layer):
    def __init__(self, channels=None, batch_norm=False, activation='relu'):
        # channels is not used here, but is included for consistency with other operations
        # batch_norm and activation are not used here, but are included for consistency with other operations
        super(AvgPool3x3Operation, self).__init__()
        self.pool = layers.AvgPool2D(3, strides=1, padding='same')
    
    def call(self, x):
        x = self.pool(x)
        return x

# Identity connection 
class IdentityOperation(layers.Layer):
    def __init__(self, channels=None, batch_norm=False, activation='relu'):
        # channels is not used here, but is included for consistency with other operations
        # batch_norm and activation are not used here, but are included for consistency with other operations
        super(IdentityOperation, self).__init__()
    
    def call(self, x):
        return x

# zeroize connection
class ZeroizeOperation(layers.Layer):
    def __init__(self, channels=None, batch_norm=False, activation='relu'):
        # channels is not used here, but is included for consistency with other operations
        # batch_norm and activation are not used here, but are included for consistency with other operations
        super(ZeroizeOperation, self).__init__()
    
    def call(self, x):
        return tf.zeros_like(x)

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Generic Cell
#--------------------------------------------------------------------------------------#
########################################################################################

operation_dict = {
    'conv3x3': Conv3x3Operation,
    'conv5x5': Conv5x5Operation,
    'sepconv3x3': SepConv3x3Operation,
    'sepconv5x5': SepConv5x5Operation,
    'conv1x1': Conv1x1Operation,
    'maxpool3x3': MaxPool3x3Operation,
    'avgpool3x3': AvgPool3x3Operation,
    'identity': IdentityOperation,
    'zeroize': ZeroizeOperation,
    'projection': ProjectionOperation,
}

def num_node_operations(nodes):
    by_node = [nodes-i-1 for i in range(nodes-1)]
    return by_node

# Generic graph cell
def make_graph_cell(nodes, operations, channels=16, batch_norm=False, activation='relu', **kwargs):
    class GraphCell(layers.Layer):
        # Graph cell connects node i to all nodes j where j>i and sums at every node
        def __init__(self, nodes, operations, channels, batch_norm, activation, **kwargs):
            super(GraphCell, self).__init__()
            # nodes is the number of nodes where operations are summed in the cell with node 1 being the input tensor
            # operations is a list of lists operations corresponding to each node
            assert nodes >= 2 # 'nodes must be greater than or equal to 2'
            self.nodes = nodes
            by_node = num_node_operations(nodes)
            self.by_node = by_node
            # create operations
            for i in range(nodes-1):
                for j in range(by_node[i]):
                    setattr(self, f'operation_{i}_{j}', operation_dict[operations[i][j]](channels, batch_norm, activation, **kwargs))

        def call(self, x):
            node_x = [x] + [tf.zeros_like(x) for _ in range(self.nodes-1)]
            for i in range(self.nodes-1):
                for j in range(self.by_node[i]):
                    node_x[i+1] += getattr(self, f'operation_{i}_{j}')(node_x[i])
            return node_x[-1]
    return GraphCell(nodes, operations, channels, batch_norm, activation)

# Generic parallel cell with concatenation of outputs
def make_parallel_concat_cell(nodes, operations, channels=16, batch_norm=False, activation='relu', **kwargs):
    class ParallelConcatCell(layers.Layer):
        # Parallel cell has one parallel track per node and concatenates the output at the end
        def __init__(self, nodes, operations, channels, batch_norm, activation, **kwargs):
            super(ParallelConcatCell, self).__init__()
            # Nodes is the number of parallel tracks
            # Operations is a list of lists corresponding to the operations at each node
            assert len(operations) == nodes
            self.nodes = nodes
            self.channels = [channels//nodes for _ in range(nodes)]
            for i in range(channels % nodes):
                self.channels[i] += 1
            self.by_node = [len(operations[i]) for i in range(nodes)]
            # create operations 
            for i in range(nodes):
                for j in range(self.by_node[i]):
                    setattr(self, f'operation_{i}_{j}', operation_dict[operations[i][j]](self.channels[i], batch_norm, activation, **kwargs))

        def call(self, x):
            node_x = [x for _ in range(self.nodes)]
            for i in range(self.nodes):
                for j in range(self.by_node[i]):
                    node_x[i] = getattr(self, f'operation_{i}_{j}')(node_x[i])
            return tf.concat(node_x, axis=-1)
    return ParallelConcatCell(nodes, operations, channels, batch_norm, activation)

# Generic parallel cell with addition of outputs
def make_parallel_add_cell(nodes, operations, channels=16, batch_norm=False, activation='relu', **kwargs):
    class ParallelAddCell(layers.Layer):
        # Parallel cell has one parallel track per node and addss the output at the end
        def __init__(self, nodes, operations, channels, batch_norm, activation, **kwargs):
            super(ParallelAddCell, self).__init__()
            # Nodes is the number of parallel tracks
            # Operations is a list of lists corresponding to the operations at each node
            assert len(operations) == nodes
            self.nodes = nodes
            self.by_node = [len(operations[i]) for i in range(nodes)]
            # create operations 
            for i in range(nodes):
                for j in range(self.by_node[i]):
                    setattr(self, f'operation_{i}_{j}', operation_dict[operations[i][j]](channels, batch_norm, activation, **kwargs))

        def call(self, x):
            node_x = [x for _ in range(self.nodes)]
            for i in range(self.nodes):
                for j in range(self.by_node[i]):
                    node_x[i] = getattr(self, f'operation_{i}_{j}')(node_x[i])
            return tf.add_n(node_x)
    return ParallelAddCell(nodes, operations, channels, batch_norm, activation)

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Macroarchitecture Cells
#--------------------------------------------------------------------------------------#
########################################################################################

# Downsample cell 
class DownsampleCell(layers.Layer):
    def __init__(self, out_filters=128, activation='relu', **kwargs):
        super(DownsampleCell, self).__init__() 
        if activation in activation_dict:
            act = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
        self.pool = layers.MaxPool2D(2, strides=2, padding='valid')
        self.conv1x1 = layers.Conv2D(out_filters, 1, strides=1, padding='same', activation=act, **kwargs)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.conv1x1(x)
        return x

# Convolution Stem 
class ConvStem(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu', **kwargs):
        super(ConvStem, self).__init__() 
        self.conv3x3 = layers.Conv2D(filters, 3, strides=2, padding='same', activation='linear', **kwargs)
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv3x3(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Final Classifier 
class FinalClassifier(layers.Layer):
    def __init__(self, num_classes=100, activation='softmax', **kwargs):
        super(FinalClassifier, self).__init__() 
        if activation in activation_dict:
            act = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation=act, **kwargs)

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        return x

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Standard Convolution Cells
#--------------------------------------------------------------------------------------#
########################################################################################

# Simple 3x3 convolution cell 
class Conv3x3Cell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(Conv3x3Cell, self).__init__()
        self.conv = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Simple 5x5 convolution cell
class Conv5x5Cell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(Conv5x5Cell, self).__init__()
        self.conv = layers.Conv2D(filters, 5, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Separable 3x3 convolution cell
class SepConv3x3Cell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(SepConv3x3Cell, self).__init__()
        self.conv = layers.SeparableConv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = tf.nn.relu(x)
        return x

# Separable 5x5 convolution cell
class SepConv5x5Cell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(SepConv5x5Cell, self).__init__()
        self.conv = layers.SeparableConv2D(filters, 5, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Convolution with Pooling Cells
#--------------------------------------------------------------------------------------#
########################################################################################

# 3x3 convolution with 2x2 max pooling cell
class Conv3x3PoolCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(Conv3x3PoolCell, self).__init__()
        self.conv = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.pool = layers.MaxPool2D(2, strides=1, padding='same')
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

# 5x5 convolution with 2x2 max pooling cell
class Conv5x5PoolCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(Conv5x5PoolCell, self).__init__()
        self.conv = layers.Conv2D(filters, 5, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.pool = layers.MaxPool2D(2, strides=1, padding='same')
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

# Separable 3x3 convolution with 2x2 max pooling cell
class SepConv3x3PoolCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(SepConv3x3PoolCell, self).__init__()
        self.conv = layers.SeparableConv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.pool = layers.MaxPool2D(2, strides=1, padding='same')
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

# Separable 5x5 convolution with 2x2 max pooling cell
class SepConv5x5PoolCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(SepConv5x5PoolCell, self).__init__()
        self.conv = layers.SeparableConv2D(filters, 5, strides=1, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.pool = layers.MaxPool2D(2, strides=1, padding='same')
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Parallel Operation Cells
#--------------------------------------------------------------------------------------#
########################################################################################

# Inception cell
class InceptionCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(InceptionCell, self).__init__()
        channels = [filters//4 for _ in range(4)]
        remainder = filters % 4
        for i in range(remainder):
            channels[i] += 1
        self.layer1 = layers.Conv2D(channels[0], 1, strides=1, padding='same', activation='linear')
        self.layer2_1 = layers.Conv2D(channels[1], 1, strides=1, padding='same', activation='relu')
        self.layer2_2 = layers.Conv2D(channels[1], 3, strides=1, padding='same', activation='linear')
        self.layer3_1 = layers.Conv2D(channels[2], 1, strides=1, padding='same', activation='relu')
        self.layer3_2 = layers.Conv2D(channels[2], 5, strides=1, padding='same', activation='linear')
        self.layer4_1 = layers.MaxPool2D(3, strides=1, padding='same')
        self.layer4_2 = layers.Conv2D(channels[3], 1, strides=1, padding='same', activation='relu')
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm3 = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
    
    def call(self, inputs):
        x1 = self.layer1(inputs)
        if self.batch_norm1 is not None:
            x1 = self.batch_norm1(x1)
        x1 = self.activation(x1)
        x2 = self.layer2_1(inputs)
        x2 = self.layer2_2(x2)
        if self.batch_norm2 is not None:
            x2 = self.batch_norm2(x2)
        x2 = self.activation(x2)
        x3 = self.layer3_1(inputs)
        x3 = self.layer3_2(x3)
        if self.batch_norm3 is not None:
            x3 = self.batch_norm3(x3)
        x3 = self.activation(x3)
        x4 = self.layer4_1(inputs)
        x4 = self.layer4_2(x4)
        x = tf.concat([x1, x2, x3, x4], axis=-1)
        return x

# Standard ResNet cell 
class ResNetCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(ResNetCell, self).__init__()
        self.layer1 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.layer2 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
    
    def call(self, inputs):
        x = self.layer1(inputs)
        if self.batch_norm1 is not None:
            x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.layer2(x)
        if self.batch_norm2 is not None:
            x = self.batch_norm2(x)
        x = tf.add(x, inputs)
        x = self.activation(x)
        return x

# ResNet XT Cell
class ResNetXTCell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(ResNetXTCell, self).__init__()
        subchannels = filters //4 + 1
        self.layer1 = layers.Conv2D(subchannels, 1, strides=1, padding='same', activation='linear')
        self.layer2 = layers.Conv2D(subchannels, 3, strides=1, padding='same', activation='linear')
        self.layer3 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='linear')
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm3 = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
    
    def call(self, inputs):
        x = self.layer1(inputs)
        if self.batch_norm1 is not None:
            x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.layer2(x)
        if self.batch_norm2 is not None:
            x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.layer3(x)
        if self.batch_norm3 is not None:
            x = self.batch_norm3(x)
        x = tf.add(x, inputs)
        x = self.activation(x)
        return x

########################################################################################
#--------------------------------------------------------------------------------------#
#                        NASBench Best Cells
#--------------------------------------------------------------------------------------#
########################################################################################

# NASBench 101 cell 
class NASBench101Cell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(NASBench101Cell, self).__init__() 
        self.proj1 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='relu')
        self.proj2 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='relu')
        self.proj3 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='relu')
        self.proj4 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='relu') 
        self.conv3x3_1 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.conv3x3_2 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.conv3x3_3 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.pool = layers.MaxPool2D(3, strides=1, padding='same')
        self.conv1x1 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='linear')
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm3 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm4 = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        # First branch
        x1 = self.proj1(inputs)
        x1 = self.conv3x3_1(x1)
        if self.batch_norm1 is not None:
            x1 = self.batch_norm1(x1)
        x1 = self.activation(x1)
        x1 = self.pool(x1)
        x1 = self.conv3x3_2(x1) 
        if self.batch_norm2 is not None:
            x1 = self.batch_norm2(x1)
        x1 = self.activation(x1)
        # Second Branch
        x2 = self.proj2(inputs)
        x2 = self.conv1x1(x2)
        if self.batch_norm3 is not None:
            x2 = self.batch_norm3(x2)
        x2 = self.activation(x2)
        # Third Branch 
        x3 = self.proj3(inputs) 
        # Fourth Branch 
        x4 = self.proj4(inputs) 

        # Combine Branches
        x123 = tf.add(x1, x2)
        x123 = tf.add(x123, x3)
        x_comb = self.conv3x3_3(x123)
        if self.batch_norm4 is not None:
            x_comb = self.batch_norm4(x_comb)
        x_comb = self.activation(x_comb)
        x_out = tf.add(x_comb, x4)
        return x_out

# NASBench 201 cell
class NASBench201Cell(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(NASBench201Cell, self).__init__() 
        self.conv3x3_1 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.conv3x3_2 = layers.Conv2D(filters, 3, strides=1, padding='same', activation='linear')
        self.conv1x1 = layers.Conv2D(filters, 1, strides=1, padding='same', activation='linear')
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm3 = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        # First branch 
        x1 = self.conv3x3_1(inputs) 
        if self.batch_norm1 is not None:
            x1 = self.batch_norm1(x1)
        x1 = self.activation(x1)
        # Second Branch
        x2 = self.conv1x1(inputs)
        if self.batch_norm2 is not None:
            x2 = self.batch_norm2(x2)
        x2 = self.activation(x2)

        # Combine Branches
        x12 = tf.add(x1, x2)
        x_comb = self.conv3x3_2(x12)
        if self.batch_norm3 is not None:
            x_comb = self.batch_norm3(x_comb)
        x_comb = self.activation(x_comb)
        x_out = tf.add(x_comb, inputs)
        return x_out

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Macroarchitecture Cells
#--------------------------------------------------------------------------------------#
########################################################################################

# Downsample cell 
class DownsampleCell(layers.Layer):
    def __init__(self, out_filters=128, activation='relu'):
        super(DownsampleCell, self).__init__() 
        if activation in activation_dict:
            act = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
        self.pool = layers.MaxPool2D(2, strides=2, padding='valid')
        self.conv1x1 = layers.Conv2D(out_filters, 1, strides=1, padding='same', activation=act)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.conv1x1(x)
        return x

# Convolution Stem 
class ConvStem(layers.Layer):
    def __init__(self, filters=128, batch_norm=False, activation='relu'):
        super(ConvStem, self).__init__() 
        self.conv3x3 = layers.Conv2D(filters, 3, strides=2, padding='same', activation='linear')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        if activation in activation_dict:
            self.activation = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')

    def call(self, inputs):
        x = self.conv3x3(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Final Classifier 
class FinalClassifier(layers.Layer):
    def __init__(self, num_classes=100, activation='softmax'):
        super(FinalClassifier, self).__init__() 
        if activation in activation_dict:
            act = activation_dict[activation]
        else:
            raise ValueError('Activation function not supported.')
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation=act)

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        return x