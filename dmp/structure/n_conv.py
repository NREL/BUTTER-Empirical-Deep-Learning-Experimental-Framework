from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from dmp.structure.n_neuron_layer import NNeuronLayer

from dmp.structure.network_module import NetworkModule
import math


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NSpatitialOperation(NNeuronLayer):
    stride: Sequence[int] = (1, 1)
    padding: str = 'same'

    @property
    def dimension(self) -> int:
        return len(self.stride)

    @property
    def num_input_channels(self) -> int:
        dimension = self.dimension
        return \
            sum((max(1, sum(i.output_shape[dimension:]))
            for i in self.inputs))


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConv(NSpatitialOperation):
    filters: int = 16
    kernel_size: Sequence[int] = (3, 3)
    batch_norm: str = 'none'

    @property
    def output_shape(self) -> Sequence[int]:
        input_conv_shape = self.input_shape[self.dimension:]
        output_conv_shape = input_conv_shape

        padding = self.padding
        if padding == 'same':
            pass
        elif padding == 'valid':
            output_conv_shape = \
                (max(0, input_dim - kernel_dim + 1)
                for input_dim, kernel_dim in zip(input_conv_shape, self.kernel_size))
        else:
            raise NotImplementedError(f'Unsupported padding method {padding}.')

        return (*output_conv_shape, self.filters)

    @property
    def num_nodes_per_filter(self) -> int:
        return math.prod(self.kernel_size)

    @property
    def num_free_parameters_in_module(self) -> int:
        num_nodes = self.num_nodes_per_filter * self.filters
        params_per_node = \
            self.num_input_channels + \
            (1 if self.use_bias else 0)
        return num_nodes * params_per_node


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NSepConv(NConv):

    @property
    def num_nodes_per_filter(self) -> int:
        return sum(self.kernel_size)


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NMaxPool(NSpatitialOperation):
    pool_size: Sequence[int] = (2, 2)

    @property
    def output_shape(self) -> Sequence[int]:
        dimension = self.dimension
        input_conv_shape = self.input_shape[dimension:]
        output_conv_shape = input_conv_shape
        
        delta = self.pool_size
        padding = self.padding
        if padding == 'same':
            delta = (1) * dimension
        elif padding == 'valid':
            pass
        else:
            raise NotImplementedError(f'Unsupported padding method {padding}.')

        output_conv_shape = (\
                int(math.floor((i - d) / s)) + 1
                for i, d, s in zip(input_conv_shape,
                delta,
                self.stride))

        return (*output_conv_shape, self.num_input_channels)


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NGlobalPool(NSpatitialOperation):
    keepdims : bool = False

    @property
    def output_shape(self) -> Sequence[int]:
        if self.keepdims:
            return (self.num_input_channels)
        else:
            pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NIdentity(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NZeroize(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConcat(NetworkModule):
    pass