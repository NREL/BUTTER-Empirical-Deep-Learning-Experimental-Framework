from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo

from dmp.layer.layer import *


@dataclass
class CNNStackAndDownsample(ModelSpec):
    num_stacks: int
    cells_per_stack: int
    stem: LayerFactory
    
    cell: LayerFactory
    #   cell_operations: List[List[str]]  #(and/or preset operations name?)
    #   conv_layer
    #   pooling_layer

    downsample:LayerFactory
    #   downsample_conv: dict
    #   downsample_pooling: dict

    output: LayerFactory


    def make_network(self) -> NetworkInfo:
        make_cnn_network(
            input_shape,
            self.num
        )

@dataclass
class ConvCellFactory(LayerFactory):

    # width: int  # width of input and output
    operations: List[List[str]]  # defines the cell structure
    conv_config: Dict[str, Any]  # conv layer configuration
    pooling_config: Dict[str, Any]  # pooling layer configuration

    def make_layer(self, inputs: List['Layer']) -> 'Layer':
        # + first serial layer is the input
        # + each serial layer is the sum of operations applied to the previous serial layers
        # + last serial layer is the output
        # + operations should be triangle structured: [[op], [op, op], [op,op,op], ...]
        serial_layers: List[Layer] = [inputs[0]]
        for cell_layer_operations in self.operations:
            parallel_operation_layers = []
            for input_layer, operation in zip(serial_layers,
                                            cell_layer_operations):
                if operation == 'zeroize':
                    continue  # skip 'zeroize' operations
                parallel_operation_layers.append(
                    self.make_layer_from_operation(
                        operation,
                        self.width,
                        conv_config,
                        pooling_config,
                        input_layer,
                    ))
            serial_layers.append(Add({}, parallel_operation_layers))
        return serial_layers[-1]

    def make_layer_from_operation(
        self,
        op: str,
        width: int,
        conv_config: Dict[str, Any],
        pooling_config: Dict[str, Any],
        input: Union[Layer, List[Layer]],
    ):
        if op == 'conv1x1':
            return DenseConv.make(\
                width, (1,1), (1,1), conv_config, input)
        elif op == 'conv3x3':
            return DenseConv.make(\
                width, (3,3), (1,1), conv_config, input)
        elif op == 'conv5x5':
            return DenseConv.make(\
                width, (5,5), (1,1), conv_config, input)
        elif op == 'sepconv3x3':
            return SeparableConv.make(\
                width, (3,3), (1,1), conv_config, input)
        elif op == 'sepconv5x5':
            return SeparableConv.make(\
                width, (5,5), (1,1), conv_config, input)
        elif op == 'maxpool3x3':
            return APoolingLayer.make(\
                MaxPool, (3, 3), (1, 1), pooling_config, input)
        elif op == 'identity':
            return IdentityOperation({}, input)
        elif op == 'zeroize':
            return ZeroizeOperation({}, input)

        raise ValueError(f'Unknown operation {op}')
            

def make_cnn_network(
    input_shape: Sequence[int],
    num_stacks: int,
    cells_per_stack: int,
    stem_factory,
    cell_factory,
    downsample_factory,
    pooling_factory,
    output_factory,
) -> Layer:
    '''
    + Structure:
        + Stem: 3x3 Conv with input activation
        + Repeat N times:
            + Repeat M times:
                + Cell
            + Downsample and double channels, optionally with Residual Connection
                + Residual: 2x2 average pooling layer with stride 2 and a 1x1
        + Global Pooling Layer
        + Dense Output Layer with output activation

    + Total depth (layer-wise or stage-wise)
    + Total number of cells
    + Total number of downsample steps
    + Number of Cells per downsample
    + Width / num channels
        + Width profile (non-rectangular widths)
    + Cell choice
    + Downsample choice (residual mode, pooling type)

    + config code inputs:
        + stem factory?
        + cell factory
        + downsample factory
        + pooling factory
        + output factory?
    '''

    layer: Layer = Input({'shape': input_shape}, [])
    layer = stem_factory(layer)
    for s in range(num_stacks):
        for c in range(cells_per_stack):
            layer = cell_factory(layer)
        layer = downsample_factory(layer)
    layer = pooling_factory(layer)
    layer = output_factory(layer)
    return layer

def stem_generator(
    filters: int,
    conv_config: Dict[str, Any],
) -> Callable[[Layer], Layer]:
    return lambda input: DenseConv.make(filters, (3, 3),
                                        (1, 1), conv_config, input)


def cell_generator(
        width: int,  # width of input and output
        operations: List[List[str]],  # defines the cell structure
        conv_config: Dict[str, Any],  # conv layer configuration
        pooling_config: Dict[str, Any],  # pooling layer configuration
) -> Callable[[Layer], Layer]:
    return lambda input: make_graph_cell(width, operations, conv_config,
                                         pooling_config, input)


def residual_downsample_generator(
    conv_config: Dict[str, Any],
    pooling_config: Dict[str, Any],
    width: int,
    stride: Sequence[int],
) -> Callable[[Layer], Layer]:

    def factory(input: Layer) -> Layer:
        long = DenseConv.make(width, (3, 3), stride, conv_config, input)
        long = DenseConv.make(width, (3, 3), (1, 1), conv_config, long)
        short = APoolingLayer.make(\
            AvgPool, (2, 2), (2, 2), pooling_config, input)
        return Add({}, [long, short])

    return factory


def output_factory_generator(
    config: Dict[str, Any],
    num_outputs: int,
) -> Callable[[Layer], Layer]:
    return lambda input: Dense(config, GlobalAveragePooling({}, input))
