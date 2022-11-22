from typing import TypeAlias, Any, Callable, Dict, Generic, Iterable, Iterator, List, Set, Sequence, Tuple, TypeVar, Type
from abc import ABC, abstractmethod

from dmp.structure.layer import Layer

ReturnType = TypeVar('ReturnType')


class LayerVisitor(ABC, Generic[ReturnType]):

    _layer_dispatch_table : Dict[str, Callable] = {
        'Input':
        lambda v, t, *args, **kwargs: \
            v._visit_Input(t, *args, **kwargs),
        'Add':
        lambda v, t, *args, **kwargs: \
            v._visit_Add(t, *args, **kwargs),
        'Concatenate':
        lambda v, t, *args, **kwargs: \
            v._visit_Concatenate(t, *args, **kwargs),
        'DenseConvolutionalLayer':
        lambda v, t, *args, **kwargs: \
            v._visit_DenseConvolutionalLayer(t, *args, **kwargs),
        'SeparableConvolutionalLayer':
        lambda v, t, *args, **kwargs: \
            v._visit_SeparableConvolutionalLayer(t, *args, **kwargs),
        'MaxPool':
        lambda v, t, *args, **kwargs:  \
            v._visit_MaxPool(t, *args, **kwargs),
        'GlobalAveragePooling':
        lambda v, t, *args, **kwargs:  \
            v._visit_GlobalAveragePooling(t, *args, **kwargs),
        'IdentityOperation':
        lambda v, t, *args, **kwargs:  \
            v._visit_IdentityOperation(t, *args, **kwargs),
        'ZeroizeOperation':
        lambda v, t, *args, **kwargs:  \
            v._visit_ZeroizeOperation(t, *args, **kwargs),
    }

    def _visit(self, layer: Layer, *args, **kwargs) -> ReturnType:
        try:
            handler = LayerVisitor._layer_dispatch_table[layer.type]
        except KeyError:
            raise NotImplementedError(f'Unsupported layer type {layer.type}.')
        return handler(
            self,
            layer,
            *args,
            **kwargs,
        )

    @abstractmethod
    def _visit_Add(self, layer: Layer, *args, **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_Concatenate(self, layer: Layer, *args, **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_DenseConvolutionalLayer(self, layer: Layer, *args,
                                       **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_SeparableConvolutionalLayer(self, layer: Layer, *args,
                                           **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_MaxPool(self, layer: Layer, *args, **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_GlobalAveragePooling(self, layer: Layer, *args,
                                      **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_IdentityOperation(self, layer: Layer, *args,
                                 **kwargs) -> ReturnType:
        pass

    @abstractmethod
    def _visit_ZeroizeOperation(self, layer: Layer, *args,
                                **kwargs) -> ReturnType:
        pass

    # @abstractmethod
    # def _visit_(self, layer: Layer) -> Any:
    #     pass
