from dataclasses import dataclass
from typing import List, Sequence

from dmp.layer import *


@dataclass
class GraphCell(LayerFactory):
    operations: List[List[Layer]]  # defines the cell structure
    output: Layer  # combines parallel layers to form single output (Add, concat, etc)

    def make_layer(
        self,
        config: LayerConfig,
        inputs: Union['Layer', List['Layer']],
    ) -> Layer:
        '''
        + first serial layer is the input
        + each serial layer is the sum of operations applied to the previous serial layers
        + last serial layer is the output
        + operations should be triangle structured: [[op], [op, op], [op,op,op], ...]
        '''
        if isinstance(inputs, Layer):
            inputs = [inputs]
            
        serial_layers: List[Layer] = inputs
        for cell_layer_operations in self.operations:
            parallel_operation_layers = []
            for input_layer, operation in zip(serial_layers, cell_layer_operations):
                if isinstance(operation, Zeroize):
                    continue  # skip 'zeroize' operations
                layer = operation.make_layer(config, [input_layer])
                parallel_operation_layers.append(layer)
            serial_layers.append(
                self.output.make_layer(config, parallel_operation_layers)
            )
        return serial_layers[-1]
