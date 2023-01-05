from abc import ABC
from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.layer import Layer

T = TypeVar('T')


class SpatitialLayer(Layer, ABC):

    def on_padding(
        self,
        on_same: Callable[[], T],
        on_valid: Callable[[], T],
    ) -> T:
        padding = self.config['padding']
        if padding == 'same':
            return on_same()
        elif padding == 'valid':
            return on_valid()
        else:
            raise NotImplementedError(f'Unsupported padding method {padding}.')

    def on_data_format(
        self,
        on_channels_last: Callable[[], T],
        on_channels_first: Callable[[], T],
    ) -> T:
        data_format = self.config['data_format']
        if data_format is None or data_format == 'channels_last':
            return on_channels_last()
        elif data_format == 'channels_first':
            return on_channels_first()
        else:
            raise NotImplementedError(
                f'Unsupported data_format {data_format}.')

    def to_conv_shape_and_channels(
        self,
        shape: Tuple,
    ) -> Tuple[Tuple, int]:
        return self.on_data_format(
            lambda: (shape[:-1], shape[-1]),
            lambda: (shape[1:] + (shape[1], )),
        )

    def to_shape(
        self,
        conv_shape: Tuple,
        num_channels: int,
    ) -> Tuple:
        return self.on_data_format(
            lambda: conv_shape + (num_channels, ),
            lambda: (num_channels, ) + conv_shape,
        )
