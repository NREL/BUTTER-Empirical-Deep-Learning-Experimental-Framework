from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, Callable
from lmarshal import Marshaler, Demarshaler
from dmp.layer.layer_factory import LayerFactory
from lmarshal.src.custom_marshalable import CustomMarshalable

network_module_types: List[Type] = []

T = TypeVar('T')

uninitialized_shape: Tuple[int, ...] = tuple()
marshaled_shape_key: str = 'shape'
marshaled_inputs_key: str = 'inputs'
marshaled_free_parameters_key: str = 'free_parameters'
empty_config: Dict[str, Any] = {}
empty_inputs: List['Layer'] = []


class Layer(LayerFactory, CustomMarshalable, ABC):

    def __init__(
        self,
        config: Dict[str, Any] = empty_config,
        input: Union['Layer', List['Layer']] = empty_inputs,
        overrides: Dict[str, Any] = empty_config,
    ) -> None:
        if not isinstance(input, List):
            input = [input]
        else:
            input = input.copy()  # defensive copy

        config = config.copy()  # defensive copy
        config.update(overrides)

        self.config: Dict[str, Any] = config
        self.inputs: List['Layer'] = input
        self.shape: Tuple[
            int, ...] = uninitialized_shape  # must be computed in context
        self.free_parameters: int = -1  # must be computed in context

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    def __copy__(self) -> 'Layer':
        return self.__class__(self.config, self.input)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key) -> bool:
        return self.config.__contains__(key)

    def make_layer(self, inputs: List['Layer']) -> 'Layer':
        result_inputs = inputs
        if len(self.inputs) > 0:
            result_inputs = [input.make_layer(inputs) for input in self.inputs]
        return self.__class__(self.config, result_inputs)

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

    @property
    def dimension(self) -> int:
        return len(self.shape) - 1

    def get(self, key, default):
        return self.config.get(key, default)

    def marshal(self, marshaler: Marshaler) -> dict:
        flat = self.config.copy()

        def safe_set(key, value):
            if key in flat:
                raise KeyError(f'Key {key} already in config.')
            flat[key] = value

        if len(self.inputs) > 0:
            safe_set(marshaled_inputs_key, self.inputs)
        if self.shape is not uninitialized_shape:
            safe_set(marshaled_shape_key, list(self.shape))
        if self.free_parameters >= 0:
            safe_set(marshaled_free_parameters_key, self.free_parameters)

        return Marshaler.marshal_dict(marshaler, flat)

    def demarshal(self, demarshaler: Demarshaler, source: dict) -> None:
        flat = Demarshaler.demarshal_dict(demarshaler, source)
        self.config = flat
        self.inputs = flat.pop(marshaled_inputs_key, [])
        self.shape = flat.pop(marshaled_shape_key, uninitialized_shape)
        self.free_parameters = flat.pop(marshaled_free_parameters_key, -1)


LayerConstructor = Callable[
    [Dict[str, Any], Union[Layer, List[Layer]], Dict[str, Any]], T]

# '''
# + single class:
#     + simple
#     + data-oriented
#     - complex abstract visitor class
#         - two things to add for each type:
#             - dispatch entry
#             - abstract method
#         - inflexible visitor interface

# + class per type
#     + clean oo
#     - many classes
#     - one class to add for each type
#     + can easily implement polymorphic methods
#     + more compact serialization
#     + could avoid serializing config or inputs in a few cases
# '''


class AElementWiseOperatorLayer(Layer):
    pass


class Dense(Layer):

    _default_config: Dict[str, Any] = {
        'activation': 'ReLu',
        'use_bias': True,
        'kernel_initializer': 'HeUniform',
        'bias_initializer': 'Zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
    }

    @staticmethod
    def make(
        units: int,
        config: Dict[str, Any] = empty_config,
        input: Union['Layer', List['Layer']] = empty_inputs,
    ) -> 'Dense':
        config['units'] = units
        return Dense(Dense._default_config, input, config)


network_module_types.append(Dense)


class Input(Layer):
    pass


network_module_types.append(Input)


class Add(AElementWiseOperatorLayer):

    @staticmethod
    def make(input: Union['Layer', List['Layer']]) -> 'Add':
        return Add({}, input)


network_module_types.append(Add)


class Concatenate(Layer):
    pass


network_module_types.append(Concatenate)


class IdentityOperation(AElementWiseOperatorLayer):
    pass


network_module_types.append(IdentityOperation)


class ZeroizeOperation(AElementWiseOperatorLayer):
    pass


network_module_types.append(ZeroizeOperation)
