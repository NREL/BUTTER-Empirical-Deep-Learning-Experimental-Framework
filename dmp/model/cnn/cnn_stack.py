from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.cnn.parallel_cell import ParallelCell

from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo

from dmp.layer import *


@dataclass
class CNNStack(ModelSpec):
    '''
    High-level definition of a typical stacked CNN Architecture
    input -> stem -> [M stacks of: N cells -> downsample ] -> final output layer
    '''

    num_stacks: int 
    cells_per_stack: int
    stem: Union[str, LayerFactory]
    downsample: Union[str, LayerFactory]
    cell: Union[str, LayerFactory]
    final: Union[str, LayerFactory]

    stem_width: int
    stack_width_scale_factor: float
    downsample_width_scale_factor: float
    cell_width_scale_factor: float

    def make_network(self) -> NetworkInfo:
        width = self.stem_width
        stage_widths = []

        # generate widths for each stack
        for s in range(self.num_stacks):
            cell_widths = []
            stage_widths.append(cell_widths)

            # downsample step
            width *= self.downsample_width_scale_factor
            cell_widths.append(int(round(width)))

            # cell steps
            for c in range(self.cells_per_stack):
                cell_widths.append(int(round(width)))
                width *= self.cell_width_scale_factor

            width *= self.stack_width_scale_factor
        print(f'widths: {stage_widths}')
        return CNNStacker(
            self.input,
            self.output,
            stage_widths,
            get_layer_factory(self.stem),
            get_layer_factory(self.downsample),
            get_layer_factory(self.cell),
            get_layer_factory(self.final),
        ).make_network()


_layer_factory_map = {
    'graph1':
    ParallelCell(
        [[conv_1x1()], [conv_3x3(), MaxPool.make([3, 3], [1, 1])]],
        Add(),
    ),
    # 'downsample_maxpool_2x2':
    # MaxPool.make([2, 2], [2, 2]),
    # 'downsample_avgpool_2x2':
    # AvgPool.make([2, 2], [2, 2]),
    'downsample_avgpool_2x2_residual_2x_conv_3x3':
    Add.make([
        AvgPool.make([2, 2], [2, 2]),
        conv_3x3([DenseConv.make(-1, [3, 3], [2, 2])]),
    ]),  # type: ignore
    'downsample_avgpool_2x2_residual_conv_3x3':
    Add.make([
        AvgPool.make([2, 2], [2, 2]),
        DenseConv.make(-1, [3, 3], [2, 2]),
    ]),
    'dense':
    Dense.make(-1, {
        'activation': 'relu',
        'initialization': 'HeUniform'
    }, []),
    'identity':
    Identity(),
}


def get_layer_factory(name: Union[str, LayerFactory]) -> LayerFactory:
    if isinstance(name, LayerFactory):
        return name

    factory = _layer_factory_map.get(name, None)
    if factory is None:
        raise KeyError(f'Unknown layer factory name {name}.')
    return factory


def add_size_and_stride(
    name: str,
    padding:str,
    factory,
    bounds: Tuple[int, int, int, int],
) -> None:
    min_size, max_size, min_stride, max_stride = bounds
    for i in range(min_size, max_size + 1):
        for s in range(min_stride, max_stride + 1):
            _layer_factory_map[f'{name}_{i}x{i}_{s}x{s}_{padding}'] = factory(i, s)


for padding in ('same', 'valid'):
    shared_config = {'padding': padding}
    add_size_and_stride(
        'conv',
        padding,
        lambda i, s: DenseConv.make(-1, [i, i], [s, s], shared_config),
        (1, 16, 1, 16),
    )

    add_size_and_stride(
        'sepconv',
        padding,
        lambda i, s: SeparableConv.make(-1, [i, i], [s, s], shared_config),
        (3, 16, 1, 16),
    )

    add_size_and_stride(
        'max_pool',
        padding,
        lambda i, s: MaxPool.make([i, i], [s, s], shared_config),
        (1, 16, 1, 16),
    )

    add_size_and_stride(
        'avg_pool',
        padding,
        lambda i, s: AvgPool.make([i, i], [s, s], shared_config),
        (1, 16, 1, 16),
    )

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
