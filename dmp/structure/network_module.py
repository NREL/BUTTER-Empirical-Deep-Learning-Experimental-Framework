from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Set, Sequence


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NetworkModule:
    inputs: List['NetworkModule'] = field(default_factory=list)

    # Deprecated members:
    # label: int = 0
    # shape: List[int] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    @property
    def input(self) -> 'NetworkModule':
        return self.inputs[0]

    @property
    def input_shape(self) -> Sequence[int]:
        return self.input.output_shape

    @property
    def output_shape(self) -> Sequence[int]:
        return self.input_shape

    @property
    def output_size(self) -> int:
        acc = 1
        for dim in self.output_shape:
            acc *= dim
        return acc

    @property
    def num_free_parameters_in_module(self) -> int:
        return 0

    @property
    def num_free_parameters_in_graph(self) -> int:
        return sum((n.num_free_parameters_in_module
                    for n in self.all_modules_in_graph))

    @property
    def dimension(self) -> int:
        return self.input.dimension

    @property
    def all_modules_in_graph(self) -> Iterator['NetworkModule']:
        '''
        An iterator over all modules in the graph (no duplicates)
        '''

        visited: Set[NetworkModule] = set()

        def visit(module: NetworkModule) -> Iterator['NetworkModule']:
            if module not in visited:
                visited.add(module)
                yield module
                for i in module.inputs:
                    yield from visit(i)

        yield from visit(self)
