from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from dmp.layer import *

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Cell Generators
#--------------------------------------------------------------------------------------#
########################################################################################

# def make_conv_stem(
#     input:Layer,
#     filters:int,
#     batch_norm:str,
#     activation='relu',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
# ) -> Layer:
#     module = DenseConvolutionalLayer(

#         filters=filters,
#         kernel_size=3,
#         stride=1,
#         padding='same',
#         batch_norm=batch_norm,
#         activation=activation,
#         kernel_regularizer=kernel_regularizer,
#         bias_regularizer=bias_regularizer,
#         activity_regularizer=activity_regularizer,
#         [input],
#     )
#     return module


# def make_final_classifier(
#     inputs,
#     classes=10,
#     activation='softmax',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
# ):
#     module = NGlobalPool(inputs=[
#         inputs,
#     ], )
#     module = NDense(
#         inputs=[
#             module,
#         ],
#         shape=[
#             classes,
#         ],
#         activation=activation,
#         kernel_regularizer=kernel_regularizer,
#         bias_regularizer=bias_regularizer,
#         activity_regularizer=activity_regularizer,
#     )
#     return module

########################################################################################
#--------------------------------------------------------------------------------------#
#                        Generic Cell Generators
#--------------------------------------------------------------------------------------#
########################################################################################


def make_layer_from_operation(
    config: Dict[str, Any],
    inputs: List[Layer],
    op: str,
):

    def make_conv_layer(factory, config, inputs, kernel_size):
        config['kernel_size'] = (kernel_size, kernel_size)
        config['strides'] = (1, 1)
        config['padding'] = 'same'
        return factory(config, inputs)

    if op == 'conv3x3':
        return make_conv_layer(DenseConvolutionalLayer, config, inputs, 3)
    elif op == 'conv5x5':
        return make_conv_layer(DenseConvolutionalLayer, config, inputs, 5)
    elif op == 'conv1x1':
        return make_conv_layer(DenseConvolutionalLayer, config, inputs, 1)
    elif op == 'sepconv3x3':
        return make_conv_layer(SeparableConvolutionalLayer, config, inputs, 3)
    elif op == 'sepconv5x5':
        return make_conv_layer(SeparableConvolutionalLayer, config, inputs, 5)
    elif op == 'maxpool3x3':
        config['pool_size'] = (3, 3)
        config['strides'] = (1, 1)
        config['padding'] = 'same'
        return MaxPool(config, inputs)
    elif op == 'identity':
        return IdentityOperation(config, inputs)
    elif op == 'zeroize':
        return ZeroizeOperation(config, inputs)

    raise ValueError(f'Unknown operation {op}')


def make_graph_cell(
        layer_config: Dict[str, Any],  # layer configuration
        input: Layer,  # cell input
        operations: List[List[str]],  # defines the cell structure
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
                continue # skip 'zeroize' operations
            parallel_operation_layers.append(
                make_layer_from_operation(
                    layer_config.copy(),
                    [input_layer],
                    operation,
                ))
        serial_layers.append(Add({}, parallel_operation_layers))
    return serial_layers[-1]

def make_parallel_cell(
        layer_config: Dict[str, Any],  # layer configuration
        input: Layer,  # cell input
        operations: List[List[str]],  # defines the cell structure
        output_factory: Callable[[Dict[str,Any], List[Layer]], Layer]
) -> Layer:
    # + multiple parallel paths of serial ops are applied and then combined
    parallel_outputs: List[Layer] = []
    for cell_layer_operations in operations:
        serial_layer = input
        for operation in cell_layer_operations:
            serial_layer = make_layer_from_operation(
                    layer_config.copy(),
                    [serial_layer],
                    operation,
                )
        parallel_outputs.append(serial_layer)
    return output_factory({}, parallel_outputs)

def make_parallel_concat_cell(
        layer_config: Dict[str, Any],  # layer configuration
        input: Layer,  # cell input
        operations: List[List[str]],  # defines the cell structure
) -> Layer:
    # + multiple parallel paths of serial ops are applied and then concatenated
    return make_parallel_cell(layer_config, input, operations, Concatenate)

def make_parallel_add_cell(
        layer_config: Dict[str, Any],  # layer configuration
        input: Layer,  # cell input
        operations: List[List[str]],  # defines the cell structure
) -> Layer:
    # + multiple parallel paths of serial ops are applied and then added
    return make_parallel_cell(layer_config, input, operations, Add)

