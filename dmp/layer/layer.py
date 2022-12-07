from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, Callable

network_module_types: List[Type] = []

T = TypeVar('T')


class Layer():

    def __init__(
        self,
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(input, List):
            input = [input]
        else:
            input = input.copy()  # defensive copy

        config = config.copy()  # defensive copy
        if overrides is not None:
            config.update(overrides)

        self.config: Dict[str, Any] = config
        self.inputs: List['Layer'] = input
        self.shape: Tuple[int, ...] = tuple()  # must be computed in context
        self.free_parameters: int = 0  # must be computed in context

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

    @property
    def dimension(self) -> int:
        return len(self.shape) - 1


LayerFactory = Callable[
    [Dict[str, Any], Union[Layer, List[Layer]], Optional[Dict[str, Any]]], T]

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

    @staticmethod
    def make(
        units: int,
        config: Dict[str, Any],
        input: Union['Layer', List['Layer']],
    ) -> 'Dense':
        return Dense(config, input, {'units': units})


network_module_types.append(Dense)


class Input(Layer):
    pass


network_module_types.append(Input)


class Add(AElementWiseOperatorLayer):
    pass


class Concatenate(Layer):
    pass


network_module_types.append(Concatenate)


class IdentityOperation(AElementWiseOperatorLayer):
    pass


network_module_types.append(IdentityOperation)


class ZeroizeOperation(AElementWiseOperatorLayer):
    pass


network_module_types.append(ZeroizeOperation)