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
