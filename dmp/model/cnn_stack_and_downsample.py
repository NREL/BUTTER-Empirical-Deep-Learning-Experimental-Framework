from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo

from dmp.layer import *


@dataclass
class CNNStacker(ModelSpec):
    num_stacks: int
    cells_per_stack: int
    stem: LayerFactory  # DenseConv, (3,3), (1,1)
    cell: LayerFactory  # ParallelCell or GraphCell
    downsample: LayerFactory
    pooling: LayerFactory

    # output: LayerFactory

    def make_network(self) -> NetworkInfo:
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

        # layer: Layer = Input({'shape': self.input_shape}, [])
        layer = self.stem.make_layer([self.input])  # type: ignore
        for s in range(self.num_stacks):
            for c in range(self.cells_per_stack):
                layer = self.cell.make_layer([layer])
            layer = self.downsample.make_layer([layer])
        layer = self.pooling.make_layer([layer])
        layer = self.output.make_layer([layer])  # type: ignore
        return NetworkInfo(layer, {})


@dataclass
class ParallelCell(LayerFactory):
    # width: int  # width of input and output
    operations: List[List[LayerFactory]]  # defines the cell structure
    output: LayerFactory  # combines parallel layers to form single output (Add, concat, etc)

    def make_layer(self, inputs: List[Layer]) -> Layer:
        # + multiple parallel paths of serial ops are applied and then combined
        parallel_outputs: List[Layer] = []
        for serial_operations in self.operations:
            serial_layer = inputs[0]
            for operation in serial_operations:
                serial_layer = operation.make_layer([serial_layer])
                serial_layer['width'] = width
            parallel_outputs.append(serial_layer)
        return self.output.make_layer(parallel_outputs)


@dataclass
class GraphCell(LayerFactory):
    # width: int  # width of input and output
    operations: List[List[LayerFactory]]  # defines the cell structure
    output: LayerFactory  # combines parallel layers to form single output (Add, concat, etc)

    def make_layer(self, inputs: List[Layer]) -> Layer:
        # + first serial layer is the input
        # + each serial layer is the sum of operations applied to the previous serial layers
        # + last serial layer is the output
        # + operations should be triangle structured: [[op], [op, op], [op,op,op], ...]
        serial_layers: List[Layer] = inputs
        for cell_layer_operations in self.operations:
            parallel_operation_layers = []
            for input_layer, operation in zip(serial_layers,
                                              cell_layer_operations):
                if isinstance(operation, ZeroizeOperation):
                    continue  # skip 'zeroize' operations
                layer = operation.make_layer([input_layer])
                layer['width'] = width
                parallel_operation_layers.append(layer)
            serial_layers.append(
                self.output.make_layer(parallel_operation_layers))
        return serial_layers[-1]


@dataclass
class ResidualDownsample(LayerFactory):
    stride: List[int]
    pooling: LayerFactory

    def make_layer(self, inputs: List[Layer]) -> Layer:
        return inputs[0]


@dataclass
class CNNStack(ModelSpec):
    num_stacks: int
    cells_per_stack: int
    # downsample_ratio: int

    stem: str
    downsample: str
    cell: str
    pooling: str

    # output: str

    def make_network(self) -> NetworkInfo:
        return CNNStacker(
            self.input,
            self.output,
            self.num_stacks,
            self.cells_per_stack,
            # self.downsample_ratio,
            layer_factory_map[self.stem],
            layer_factory_map[self.downsample],
            layer_factory_map[self.cell],
            layer_factory_map[self.pooling],
        ).make_network()


layer_factory_map = {
    'graph1':
    ParallelCell(
        [[conv1x1()], [conv3x3(), max_pool([3, 3], [1, 1], {}, [])]],
        Add({}, []),
    ),
    'downsample_residual_1':
    Add({}, [
        APoolingLayer.make(AvgPool, (2, 2), (2, 2), pooling_config, input),
        DenseConv.make(
            width, (3, 3), (1, 1), conv_config,
            DenseConv.make(width, (3, 3), stride, conv_config, input))
    ]),
    'dense':
    Dense.make(-1, {
        'activation': 'relu',
        'initialization': 'HeUniform'
    }, []),
}


def add_size_and_stride(name: str, factory, min_size: int, max_size: int,
                        min_stride: int, max_stride: int) -> None:
    for i in range(min_size, max_size + 1):
        for s in range(min_stride, max_stride + 1):
            layer_factory_map[f'{name}{i}x{i}_{s}x{s}'] = factory(i, s)


add_size_and_stride('conv',
                    lambda i, s: DenseConv.make(-1, [i, i], [s, s], {}, []), 1,
                    16, 1, 16)
add_size_and_stride(
    'sepconv', lambda i, s: SeparableConv.make(-1, [i, i], [s, s], {}, []), 1,
    16, 1, 16)
add_size_and_stride('max_pool',
                    lambda i, s: MaxPool.make([i, i], [s, s], {}, []), 1, 16,
                    1, 16)
add_size_and_stride('avg_pool',
                    lambda i, s: AvgPool.make([i, i], [s, s], {}, []), 1, 16,
                    1, 16)

# Add({}, [
#     DenseConv.make(-1, [3, 3], [1, 1], {}, []),
#     MaxPool.make([3, 3], [1, 1], {},
#                  [DenseConv.make(-1, [1, 1], [1, 1], {}, [])])
# ]),
# 'graph2':
# Add({}, [
#     DenseConv.make(-1, [3, 3], [1, 1], {}, []),
#     MaxPool.make([3, 3], [1, 1], {},
#                  [DenseConv.make(-1, [1, 1], [1, 1], {}, [])])
# ])

# def make_layer_from_operation(
#         self,
#         op: LayerFactory,
#         width: int,
#         conv_config: Dict[str, Any],
#         pooling_config: Dict[str, Any],
#         input: Union[Layer, List[Layer]],
#     ):
#         if op == 'conv1x1':
#             return DenseConv.make(\
#                 width, (1,1), (1,1), conv_config, input)
#         elif op == 'conv3x3':
#             return DenseConv.make(\
#                 width, (3,3), (1,1), conv_config, input)
#         elif op == 'conv5x5':
#             return DenseConv.make(\
#                 width, (5,5), (1,1), conv_config, input)
#         elif op == 'sepconv3x3':
#             return SeparableConv.make(\
#                 width, (3,3), (1,1), conv_config, input)
#         elif op == 'sepconv5x5':
#             return SeparableConv.make(\
#                 width, (5,5), (1,1), conv_config, input)
#         elif op == 'maxpool3x3':
#             return APoolingLayer.make(\
#                 MaxPool, (3, 3), (1, 1), pooling_config, input)
#         elif op == 'identity':
#             return IdentityOperation({}, input)
#         elif op == 'zeroize':
#             return ZeroizeOperation({}, input)

#         raise ValueError(f'Unknown operation {op}')

# def stem_generator(
#     filters: int,
#     conv_config: Dict[str, Any],
# ) -> Callable[[Layer], Layer]:
#     return lambda input: DenseConv.make(filters, (3, 3),
#                                         (1, 1), conv_config, input)

# def cell_generator(
#         width: int,  # width of input and output
#         operations: List[List[str]],  # defines the cell structure
#         conv_config: Dict[str, Any],  # conv layer configuration
#         pooling_config: Dict[str, Any],  # pooling layer configuration
# ) -> Callable[[Layer], Layer]:
#     return lambda input: make_graph_cell(width, operations, conv_config,
#                                          pooling_config, input)

# def residual_downsample_generator(
#     conv_config: Dict[str, Any],
#     pooling_config: Dict[str, Any],
#     width: int,
#     stride: Sequence[int],
# ) -> Callable[[Layer], Layer]:

#     def factory(input: Layer) -> Layer:
#         long = DenseConv.make(width, (3, 3), stride, conv_config, input)
#         long = DenseConv.make(width, (3, 3), (1, 1), conv_config, long)
#         short = APoolingLayer.make(\
#             AvgPool, (2, 2), (2, 2), pooling_config, input)
#         return Add({}, [long, short])

#     return factory

# def output_factory_generator(
#     config: Dict[str, Any],
#     num_outputs: int,
# ) -> Callable[[Layer], Layer]:
#     return lambda input: Dense(config, GlobalAveragePooling({}, input))
