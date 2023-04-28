from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    Callable,
)
from lmarshal.src.custom_marshalable import CustomMarshalable


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

'''
+ how to make keras layer (factor into visitor)
+ compute shape
+ count free parameters
+ convience constructors/factories

'''

# def register_layer_type(type: Type) -> None:
#     from dmp.marshal_registry import register_type
#     register_type(type)
#     # layer_types.append(type)


class LayerFactory(ABC):
    '''
    Factory class that makes layers.
    '''

    @abstractmethod
    def make_layer(
        self,
        config: 'LayerConfig',
        inputs: Union['Layer', List['Layer']],
    ) -> 'Layer':
        '''
        Factory method that makes a layer that consumes the provided inputs and overrides layer configurations with the
        provided config dictionary.
        '''
        pass


class Layer(LayerFactory, CustomMarshalable, ABC):
    '''
    Defines a network Layer. Close to 1-1 correspondance with Keras layer classes.
    '''

    def __init__(
        self,
        config: LayerConfig = empty_config,  # keras constructor kwargs
        input: Union['Layer', List['Layer']] = empty_inputs,  # inputs to this Layer
        overrides: LayerConfig = empty_config,  # optional override keys of config
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
            int, ...
        ] = uninitialized_shape  # must be computed in context
        self.free_parameters: int = (
            unitialized_parameter_count  # must be computed in context
        )

        # print(f'make layer {type(self)} {config}')

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

    def describe(self) -> str:
        graph_inputs = []
        output_map = {}
        
        for layer in self.layers_post_ordered:
            output_map.setdefault(layer, [])

            if len(layer.inputs) == 0:
                graph_inputs.append(layer)

            for input in layer.inputs:
                output_map.setdefault(input, []).append(layer)

        layer_order_index = {}
        ordered_layers = []

        def compute_order(layer):
            if layer not in layer_order_index:
                layer_order_index[layer] = len(layer_order_index)
                ordered_layers.append(layer)
                for output in output_map[layer]:
                    compute_order(output)

        for input in graph_inputs:
            compute_order(input)

        descriptions = []
        for index, layer in enumerate(ordered_layers):
            # describe this layer
            params = []            
            if 'kernel_size' in layer:
                params.append('x'.join((str(s) for s in layer['kernel_size'])))

            if 'strides' in layer:
                params.append(f's: {"x".join((str(s) for s in layer["strides"]))}')
            
            if 'units' in layer:
                params.append(f'x{layer["units"]}')
            
            if not layer.use_bias:
                params.append('-b')

            if 'activation' in layer:
                params.append(f'{layer["activation"]}')
            
            if 'name' in layer:
                params.append(f'"{layer["name"]}"')

            output_indicies = ', '.join(
                (str(layer_order_index[output]) for output in output_map[layer])
            )
            descriptions.append(f'{index}: ({type(layer)}: {layer.free_parameters}, {layer.computed_shape}, {", ".join(params)} -> [{output_indicies}])')

        return '\n'.join(descriptions)

    def update_if_exists(self, overrides: LayerConfig) -> None:
        '''
        Updates layer config with the supplied overrides only where their keys already are set in the config.
        '''
        for k, v in overrides.items():
            if k in self.config:
                self.config[k] = v

    def insert_if_not_exists(self, to_insert: LayerConfig) -> None:
        '''
        Sets items that don't already exist in the config.
        '''
        for k, v in to_insert.items():
            if k not in self.config:
                self.config[k] = v

    def update(self, overrides: LayerConfig) -> None:
        '''
        Overrides/sets the config using the supplied overrides config.
        '''
        self.config.update(overrides)

    def make_layer(
        self,
        override_if_exists: LayerConfig,
        inputs: Union['Layer', List['Layer']],
    ) -> 'Layer':
        '''
        Generates a matching layer graph and links it to the supplied inputs.
        That is, this function traverses this layer and it's descendents (inputs), making a matching copy of the graph until it reaches layers with no inputs.
        These leaf layers have their inputs set to the supplied inputs.
        When copying, the copies have update_if_exists(override_if_exists) called on them, allowing the caller to override the returned layers' configurations.
        '''
        if isinstance(inputs, Layer):
            inputs = [inputs]

        layer_inputs = inputs
        if len(self.inputs) > 0:
            layer_inputs = [
                input.make_layer(override_if_exists, inputs) for input in self.inputs
            ]

        result = self.__class__(self.config, layer_inputs)
        result.update_if_exists(override_if_exists)
        return result

    @property
    def name(self) -> Optional[str]:
        '''
        An optional name of this layer. Set in the config.
        '''
        return self.config.get('name', None)

    @property
    def input(self) -> 'Layer':
        '''
        The zeroth input. Useful shortcut for layers with only one input.
        '''
        return self.inputs[0]

    @property
    def layers(self) -> Iterator['Layer']:
        '''
        An iterator over all layers in the graph without duplicates.
        Ordering not guaranteed.
        '''
        return self.layers_pre_ordered

    @property
    def layers_pre_ordered(self) -> Iterator['Layer']:
        '''
        An iterator over all layers in the graph without duplicates.
        Pre-order traversal (this layer, the output layer, is the root),
        skipping layers already traversed.
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
    def layers_post_ordered(self) -> Iterator['Layer']:
        '''
        An iterator over all layers in the graph without duplicates.
        Post-order traversal (this layer, the output layer, is the root),
        skipping layers already traversed.
        '''
        visited: Set['Layer'] = set()

        def visit(current: 'Layer') -> Iterator['Layer']:
            if current not in visited:
                visited.add(current)
                for i in current.inputs:
                    yield from visit(i)
                yield current

        yield from visit(self)

    @property
    def leaves(self) -> Iterator['Layer']:
        '''
        An iterator over all leaf layers (layers with zero inputs) of the layer graph.
        '''

        for layer in self.layers:
            if len(layer.inputs) == 0:
                yield layer

    @property
    def use_bias(self) -> bool:
        '''
        Shortcut to get the use_bias config parameter.
        '''
        return self.config.get('use_bias', True)

    @property
    def dimension(self) -> int:
        '''
        Shortcut to get the dimensionality of the computed shape.
        '''
        return len(self.computed_shape) - 1

    def get(self, key, default):
        '''
        Returns the value matching the given key from the config, or default if the key is not set.
        '''
        return self.config.get(key, default)

    def marshal(self) -> dict:
        '''
        Custom marshalling implementation.
        Converts some tuples to lists for more readable marshaling/serialization.
        '''
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
        '''
        Custom demarshaling implementation.
        Restores tuples that were converted to lists during marshalling.
        '''
        self.config = flat
        self.inputs = flat.pop(marshaled_inputs_key, [])
        computed_shape = flat.pop(marshaled_computed_shape_key, None)
        self.computed_shape = (
            uninitialized_shape if computed_shape is None else tuple(computed_shape)
        )
        self.free_parameters = flat.pop(
            marshaled_free_parameters_key, unitialized_parameter_count
        )


LayerConstructor = Callable[[LayerConfig, Union[Layer, List[Layer]], LayerConfig], T]
