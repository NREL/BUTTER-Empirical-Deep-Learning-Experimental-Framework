from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from dmp.structure.n_neuron_layer import NNeuronLayer

from dmp.structure.network_module import NetworkModule
import math



@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NSpatitialOperation(NNeuronLayer):
    dimension: int = 2
    data_format: str = 'channels_last' # 'channels_first' or 'channels_last'

    @property
    def num_input_channels(self) -> int:
        dimension = self.dimension
        return \
            sum((max(1, sum(i.output_shape[dimension:]))
            for i in self.inputs))
    
    def _on_data_format(
        self,
        on_channels_last: Callable[[], Any],
        on_channels_first: Callable[[], Any],
    ) -> Any:
        data_format = self.data_format
        if data_format == 'channels_last':
            return on_channels_last()
        elif data_format == 'channels_first':
            return on_channels_first()
        else:
            raise NotImplementedError(f'Unsupported data_format {data_format}.')

@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NSpatitialScan(NSpatitialOperation):
    stride: Sequence[int] = (1, 1)
    padding: str = 'same' # 'same' or 'valid'

    def _on_padding(
        self,
        on_same: Callable[[], Any],
        on_valid: Callable[[], Any],
    ) -> Any:
        padding = self.padding
        if padding == 'same':
            return on_same()
        elif padding == 'valid':
            return on_valid()
        else:
            raise NotImplementedError(f'Unsupported padding method {padding}.')


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConv(NSpatitialScan):
    filters: int = 16
    kernel_size: Sequence[int] = (3, 3)
    batch_norm: str = 'none' # 'all' or 'none'

    @property
    def output_shape(self) -> Sequence[int]:
        input_conv_shape = self.input_shape[self.dimension:]
        output_conv_shape = self._on_padding(
            lambda: input_conv_shape,
            lambda: (max(0, input_dim - kernel_dim + 1) for input_dim,
                     kernel_dim in zip(input_conv_shape, self.kernel_size)),
        )

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
class NMaxPool(NSpatitialScan):
    pool_size: Sequence[int] = (2, 2)

    @property
    def output_shape(self) -> Sequence[int]:
        dimension = self.dimension
        input_conv_shape = self.input_shape[dimension:]
        output_conv_shape = input_conv_shape
        delta = self._on_padding(
            lambda: (1, ) * dimension,
            lambda: self.pool_size,
        )

        output_conv_shape = (\
                int(math.floor((i - d) / s)) + 1
                for i, d, s in zip(input_conv_shape,
                delta,
                self.stride))

        return (*output_conv_shape, self.num_input_channels)


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NGlobalPool(NSpatitialOperation):
    keepdims: bool = False

    @property
    def output_shape(self) -> Sequence[int]:
        num_channels = (self.num_input_channels,)
        if self.keepdims:
            ones = (1,) * self.dimension
            if self.data_format=='channels_last':
                return ones + num_channels
            return num_channels + ones
        return num_channels


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NIdentity(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NZeroize(NetworkModule):
    pass


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConcat(NetworkModule):
    pass