from dataclasses import dataclass
from typing import Callable, Dict, List, Iterator, Any, Set, Tuple, TypeVar
'''
+ single class:
    + simple 
    + data-oriented
    - complex abstract visitor class 
        - two things to add for each type:
            - dispatch entry
            - abstract method
        - inflexible visitor interface

+ class per type
    + clean oo 
    - many classes
    - one class to add for each type
    + can easily implement polymorphic methods
    + more compact serialization
    + could avoid serializing config or inputs in a few cases
'''
T = TypeVar('T')


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class Layer():

    config: Dict[str, Any]
    inputs: List['Layer']

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    @property
    def input(self) -> 'Layer':
        return self.inputs[0]

    @property
    def all_descendants(self) -> Iterator['Layer']:
        '''
        An iterator over all layers in the graph without duplicates.
        '''

        visited: Set['Layer'] = set()

        def visit(current: 'Layer') -> Iterator['Layer']:
            if current not in visited:
                visited.add(current)
                yield current
                for i in current.inputs:
                    yield from visit(i)

        yield from visit(self)

    @property
    def use_bias(self) -> bool:
        return self.config.get('use_bias', True)


class AElementWiseOperatorLayer(Layer):
    pass


class Dense(Layer):
    pass


class Input(Layer):
    pass


class Add(AElementWiseOperatorLayer):
    pass


class Concatenate(Layer):
    pass


class ASpatitialLayer(Layer):

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


class AConvolutionalLayer(ASpatitialLayer):
    pass


class DenseConvolutionalLayer(AConvolutionalLayer):
    pass


class SeparableConvolutionalLayer(AConvolutionalLayer):
    pass


class APoolingLayer(ASpatitialLayer):

    @property
    def strides(self) -> Tuple:
        config = self.config
        strides = config.get('strides', None)
        if strides is not None:
            return strides
        return config['pool_size']


class MaxPool(APoolingLayer):
    pass


class AvgPool(APoolingLayer):
    pass


class AGlobalPoolingLayer(ASpatitialLayer):
    pass


class GlobalAveragePooling(AGlobalPoolingLayer):
    pass


class GlobalMaxPooling(AGlobalPoolingLayer):
    pass


class IdentityOperation(AElementWiseOperatorLayer):
    pass


class ZeroizeOperation(AElementWiseOperatorLayer):
    pass


class ProjectionOperation(AConvolutionalLayer):
    pass
