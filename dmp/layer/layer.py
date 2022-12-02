from typing import Any, Dict, Iterator, List, Set, Type

network_module_types: List[Type] = []


class Layer():

    def __init__(self, config: Dict[str, Any], inputs: List['Layer']) -> None:
        self.config: Dict[str, Any] = config
        self.inputs: List['Layer'] = inputs

    # config: Dict[str, Any]
    # inputs: List['Layer']

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
    pass


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