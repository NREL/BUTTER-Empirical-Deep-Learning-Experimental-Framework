from dataclasses import dataclass
from typing import Dict, List, Iterator, Any, Set


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class Layer():

    type: str
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
        An iterator over all layers in the graph without duplicates
        '''

        visited: Set['Layer'] = set()

        def visit(current: 'Layer') -> Iterator['Layer']:
            if current not in visited:
                visited.add(current)
                yield current
                for i in current.inputs:
                    yield from visit(i)

        yield from visit(self)
