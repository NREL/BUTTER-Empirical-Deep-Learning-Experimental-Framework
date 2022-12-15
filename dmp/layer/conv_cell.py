from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from dmp.layer import *

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Generic Cell Generators
#--------------------------------------------------------------------------------------#
########################################################################################


def make_layer_from_operation(
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
        return Identity({}, input)
    elif op == 'zeroize':
        return Zeroize({}, input)

    raise ValueError(f'Unknown operation {op}')


def make_graph_cell(
        width: int,  # width of all layers, including output
        operations: List[List[str]],  # defines the cell structure
        conv_config: Dict[str, Any],  # conv layer configuration
        pooling_config: Dict[str, Any],  # pooling layer configuration
        input: Layer,  # cell input
) -> Layer:
    # + first serial layer is the input
    # + each serial layer is the sum of operations applied to the previous serial layers
    # + last serial layer is the output
    # + operations should be triangle structured: [[op], [op, op], [op,op,op], ...]
    serial_layers: List[Layer] = [input]
    for cell_layer_operations in operations:
        parallel_operation_layers = []
        for input_layer, operation in zip(serial_layers,
                                          cell_layer_operations):
            if operation == 'zeroize':
                continue  # skip 'zeroize' operations
            parallel_operation_layers.append(
                make_layer_from_operation(
                    operation,
                    width,
                    conv_config,
                    pooling_config,
                    input_layer,
                ))
        serial_layers.append(Add({}, parallel_operation_layers))
    return serial_layers[-1]


def make_parallel_cell(
        output_factory: Callable[[Dict[str, Any], List[Layer]],
                                 Layer],  # output layer factory
        width: int,  # width of input and output
        operations: List[List[str]],  # defines the cell structure
        conv_config: Dict[str, Any],  # conv layer configuration
        pooling_config: Dict[str, Any],  # pooling layer configuration
        input: Layer,  # cell input
) -> Layer:
    # + multiple parallel paths of serial ops are applied and then combined
    parallel_outputs: List[Layer] = []
    for cell_layer_operations in operations:
        serial_layer = input
        for operation in cell_layer_operations:
            serial_layer = make_layer_from_operation(
                operation,
                width,
                conv_config,
                pooling_config,
                serial_layer,
            )
        parallel_outputs.append(serial_layer)
    return output_factory({}, parallel_outputs)


def make_parallel_add_cell(
        operations: List[List[str]],  # defines the cell structure
        width: int,  # width of input and output
        conv_config: Dict[str, Any],  # conv layer configuration
        pooling_config: Dict[str, Any],  # pooling layer configuration
        input: Layer,  # cell input
) -> Layer:
    # + multiple parallel paths of serial ops are applied and then added
    return make_parallel_cell(Add, width, operations, conv_config,
                              pooling_config, input)


# def make_parallel_concat_cell(
#         layer_config: Dict[str, Any],  # layer configuration
#         input: Layer,  # cell input
#         operations: List[List[str]],  # defines the cell structure
# ) -> Layer:
#     # + multiple parallel paths of serial ops are applied and then concatenated
#     return make_parallel_cell(layer_config, input, operations, Concatenate)
