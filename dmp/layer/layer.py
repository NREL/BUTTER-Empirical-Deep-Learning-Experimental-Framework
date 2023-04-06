from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, Callable
from lmarshal.src.custom_marshalable import CustomMarshalable
import dmp.keras_interface.keras_keys as keras_keys

LayerConfig = Dict[str, Any]

uninitialized_shape: Tuple[int, ...] = tuple()
marshaled_computed_shape_key: str = 'computed_shape'
marshaled_inputs_key: str = 'inputs'
marshaled_free_parameters_key: str = 'free_parameters'
empty_config: Dict[str, Any] = {}
empty_inputs: List = []
unitialized_parameter_count: int = -1

T = TypeVar('T')

# layer_types: List[Type] = []


# def register_layer_type(type: Type) -> None:
#     from dmp.marshal_registry import register_type
#     register_type(type)
#     # layer_types.append(type)

class LayerFactory(ABC):
    '''
    Thing that can make a layer.
    '''

    @abstractmethod
    def make_layer(
        self,
        inputs: List['Layer'],
        config: 'LayerConfig',
    ) -> 'Layer':
        pass


class Layer(LayerFactory, CustomMarshalable, ABC):
    '''
    Defines a network Layer. Close to 1-1 correspondance with Keras layer classes.
    '''

    def __init__(
        self,
        config: LayerConfig = empty_config, # keras constructor kwargs
        input: Union['Layer', List['Layer']] = empty_inputs, # input Layers to this Layers
        overrides: LayerConfig = empty_config, # optional override keys of config
    ) -> None:
        if not isinstance(input, List):
            input = [input]
        else:
            input = input.copy()  # defensive copy
        
        config = config.copy()  # defensive copy
        config.update(overrides)

        self.config: LayerConfig = config
        self.inputs: List['Layer'] = input
        self.computed_shape: Tuple[
            int, ...] = uninitialized_shape  # must be computed in context
        self.free_parameters: int = unitialized_parameter_count  # must be computed in context

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    def __copy__(self) -> 'Layer':
        return self.__class__(self.config, self.inputs)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key) -> bool:
        return self.config.__contains__(key)

    def update_if_exists(self, overrides: LayerConfig) -> None:
        for k, v in overrides.items():
            if k in self.config:
                self.config[k] = v

    def insert_if_not_exists(self, to_insert: LayerConfig) -> None:
        for k, v in to_insert.items():
            if k not in self.config:
                self.config[k] = v

    def update(self, overrides: LayerConfig) -> None:
        self.config.update(overrides)

    def make_layer(
        self,
        inputs: List['Layer'],
        override_if_exists: LayerConfig,
    ) -> 'Layer':
        layer_inputs = inputs
        if len(self.inputs) > 0:
            layer_inputs = [
                input.make_layer(inputs, override_if_exists)
                for input in self.inputs
            ]

        result = self.__class__(self.config, layer_inputs)
        result.update_if_exists(override_if_exists)
        return result

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
        return self.config.get(keras_keys.use_bias, True)

    @property
    def dimension(self) -> int:
        return len(self.computed_shape) - 1

    def get(self, key, default):
        return self.config.get(key, default)

    def marshal(self) -> dict:
        flat = self.config.copy()

        def safe_set(key, value):
            if key in flat:
                raise KeyError(f'Key {key} already in config.')
            flat[key] = value

        if len(self.inputs) > 0:
            safe_set(marshaled_inputs_key, self.inputs)
        if self.computed_shape is not uninitialized_shape:
            safe_set(marshaled_computed_shape_key, list(self.computed_shape))
        if self.free_parameters >= 0:
            safe_set(marshaled_free_parameters_key, self.free_parameters)
        return flat

    def demarshal(self, flat: dict) -> None:
        self.config = flat
        self.inputs = flat.pop(marshaled_inputs_key, [])
        computed_shape = flat.pop(marshaled_computed_shape_key, None)
        self.computed_shape = uninitialized_shape if computed_shape is None \
            else tuple(computed_shape)
        self.free_parameters = flat.pop(marshaled_free_parameters_key,
                                        unitialized_parameter_count)


LayerConstructor = Callable[
    [LayerConfig, Union[Layer, List[Layer]], LayerConfig], T]
