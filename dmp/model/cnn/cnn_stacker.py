from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo

from dmp.layer import *


@dataclass
class CNNStacker(ModelSpec):
    stage_widths: List[List[int]]
    stem: LayerFactory  # DenseConv, (3,3), (1,1)
    cell: LayerFactory  # ParallelCell or GraphCell
    downsample: LayerFactory
    pooling: LayerFactory

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

        layer: Layer = self.input  # type: ignore
        for stage, cell_widths in enumerate(self.stage_widths):
            for cell, cell_width in enumerate(cell_widths):
                config = {'filters': cell_width}
                if cell == 0:
                    if stage == 0:
                        layer = self.stem.make_layer([layer], config)
                    else:
                        layer = self.downsample.make_layer([layer], config)
                else:
                    layer = self.cell.make_layer([layer], config)

        layer = self.pooling.make_layer([layer], {})
        layer = self.output.make_layer([layer], {})  # type: ignore
        return NetworkInfo(layer, {})
