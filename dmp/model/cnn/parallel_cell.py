from dataclasses import dataclass
from typing import List, Sequence

from dmp.layer import *


@dataclass
class ParallelCell(LayerFactory):
    # width: int  # width of input and output
    operations: List[List[Layer]]  # defines the cell structure
    output: Layer  # combines parallel layers to form single output (Add, concat, etc)

    def make_layer(
        self,
        config: LayerConfig,
        inputs: Union['Layer', List['Layer']],
    ) -> Layer:
        '''
        + multiple parallel paths of serial ops are applied and then combined
        '''
        
        if isinstance(inputs, Layer):
            inputs = [inputs]

        parallel_outputs: List[Layer] = []
        for serial_operations in self.operations:
            serial_layer = inputs[0]
            for operation in serial_operations:
                serial_layer = operation.make_layer(config, [serial_layer])
            parallel_outputs.append(serial_layer)
        return self.output.make_layer(config, parallel_outputs)
