from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo

from dmp.layer import *


@dataclass
class CNNStacker(ModelSpec):
    '''
    Defines a typical CNN structure with sections:
        input -> stem -> [M stacks of: N cells -> downsample ] -> final output layer
    '''

    stage_widths: List[List[int]]
    stem: LayerFactory
    downsample: LayerFactory
    cell: LayerFactory
    final: LayerFactory

    def make_network(self) -> NetworkInfo:
        '''

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

        layer: Layer = self.input  # type: ignore
        for stage, cell_widths in enumerate(self.stage_widths):
            print('stage')
            for cell, cell_width in enumerate(cell_widths):
                config = {'filters': cell_width}
                if cell < len(cell_widths) - 1:
                    if stage == 0:
                        print('stem')
                        layer = self.stem.make_layer([layer], config)
                    else:
                        print('cell')
                        layer = self.cell.make_layer([layer], config)
                else:
                    print('downsample')
                    layer = self.downsample.make_layer([layer], config)

        layer = self.final.make_layer([layer], {})
        layer = self.output.make_layer([layer], {})  # type: ignore
        return NetworkInfo(layer, {})
