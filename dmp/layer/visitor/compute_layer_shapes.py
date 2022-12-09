from functools import singledispatchmethod
import math
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Sequence, Tuple, TypeVar, Union

from dmp.layer import *


_invalid_shape = tuple()

class ComputeLayerShapesVisitor:

    def __init__(self, target: Layer) -> None:
        self._visited: Set[Layer] = set()
        self._compute_output_shape(target)

    def _compute_output_shape(self, target: Layer) -> Optional[Tuple]:
        if target in self._visited:
            return

        target.shape = _invalid_shape
        self._visited.add(target)
        for i in target.inputs:
            self._compute_output_shape(i)
        target.shape = self._visit(target, target.config)

    def _get_output_shape(self, target: Layer) -> Tuple:
        self._compute_output_shape(target)
        shape = target.shape
        if shape is _invalid_shape:
            raise ValueError(f'Can not determine shape of Layer {target}.')
        return shape

    def _get_input_shape(self, target: Layer) -> Tuple:
        return self._get_output_shape(target.input)

    @singledispatchmethod
    def _visit(self, target: Layer, config: Dict) -> Tuple:
        raise NotImplementedError(f'Unsupported Layer of type {type(target)}.')

    @_visit.register
    def _(self, target: Input, config: Dict) -> Tuple:
        return config['shape']

    @_visit.register
    def _(self, target: Dense, config: Dict) -> Tuple:
        return (config['units'], )

    @_visit.register
    def _(self, target: AElementWiseOperatorLayer, config: Dict) -> Tuple:
        return self._get_input_shape(target)

    @_visit.register
    def _(self, target: Concatenate, config: Dict) -> Tuple:
        axis = config['axis']
        total = sum((self._get_output_shape(i)[axis] for i in target.inputs))
        input_shape = self._get_input_shape(target)
        return input_shape[:axis] + (total, ) + input_shape[axis + 1:]

    @_visit.register
    def _(self, target: ConvolutionalLayer, config: Dict) -> Tuple:
        input_conv_shape, input_channels = \
            target.to_conv_shape_and_channels(
                self._get_input_shape(target))
        strides = config['strides']

        output_conv_shape = tuple(
            target.on_padding(
                lambda: (math.ceil(float(d) / float(s)) \
                    for d, s in zip(
                    input_conv_shape,
                    strides,
                )),
                lambda: (math.ceil(float(d - k + 1) / float(s))\
                    for d, s, k in zip(
                        input_conv_shape,
                        strides,
                        config['kernel_size'],
                    )),
            ))

        return target.to_shape(
            output_conv_shape,
            config['filters'],
        )

    @_visit.register
    def _(self, target: APoolingLayer, config: Dict) -> Tuple:
        input_conv_shape, input_channels = \
            target.to_conv_shape_and_channels(
                self._get_input_shape(target))

        delta = target.on_padding(
            lambda: (1, ) * len(input_conv_shape),
            lambda: config['pool_size'],
        )

        output_conv_shape = tuple((int(math.floor((i - d) / s)) + 1
                                   for i, d, s in zip(
                                       input_conv_shape,
                                       delta,
                                       target.strides,
                                   )))

        return target.to_shape(
            output_conv_shape,
            input_channels,
        )

    @_visit.register
    def _(self, target: AGlobalPoolingLayer, config: Dict) -> Tuple:
        input_conv_shape, input_channels = \
            target.to_conv_shape_and_channels(
                self._get_input_shape(target))

        if config.get('keepdims', False):
            return target.to_shape(
                (1, ) * len(input_conv_shape),
                input_channels,
            )
        return (input_channels, )


def compute_layer_shapes(target: Layer) -> None:
    ComputeLayerShapesVisitor(target)