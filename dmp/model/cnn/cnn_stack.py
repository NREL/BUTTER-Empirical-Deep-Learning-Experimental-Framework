from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.cnn.parallel_cell import ParallelCell

from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo

from dmp.layer import *
import dmp.keras_interface.keras_keys as keras_keys

@dataclass
class CNNStack(ModelSpec):
    '''
    High-level definition of a typical stacked CNN Architecture
    input -> stem -> [M stacks of: N cells -> downsample ] -> final output layer
    '''

    num_stacks: int = 1
    cells_per_stack: int = 1
    stem: Union[str, LayerFactory] = 'conv_3x3_1x1_same'
    downsample: Union[str, LayerFactory]  = 'max_pool_3x3_1x1_same'
    cell: Union[str, LayerFactory] = 'conv_3x3_1x1_same'
    final: Union[str, LayerFactory] = field(default_factory=lambda:Dense.make(4096))

    stem_width: int = 64
    stack_width_scale_factor: float = 1.0
    downsample_width_scale_factor: float = 1.0
    cell_width_scale_factor: float = 1.0

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
        # conv_3x3([DenseConv.make(-1, [3, 3], [2, 2])]),
        DenseConv.make(-1, [3, 3], [1, 1], {keras_keys.padding:'same'}, [DenseConv.make(-1, [3, 3], [2, 2], {keras_keys.padding:'same'})])
    ]),  # type: ignore
    'downsample_avgpool_2x2_residual_conv_3x3':
    Add.make([
        AvgPool.make([2, 2], [2, 2]),
        DenseConv.make(-1, [3, 3], [2, 2]),
    ]),
    'dense':
    Dense.make(-1, {
        keras_keys.activation: 'relu',
        keras_keys.initialization: 'HeUniform'
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
    shared_config = {keras_keys.padding: padding}
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
