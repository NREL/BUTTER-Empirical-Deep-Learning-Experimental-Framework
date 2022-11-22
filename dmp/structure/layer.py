from dataclasses import dataclass
from typing import Callable, Dict, List, Iterator, Any, Set


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

    def on_padding(
        self,
        on_same: Callable[[], Any],
        on_valid: Callable[[], Any],
    ) -> Any:
        padding = self.config['padding']
        if padding == 'same':
            return on_same()
        elif padding == 'valid':
            return on_valid()
        else:
            raise NotImplementedError(f'Unsupported padding method {padding}.')
    
    def on_data_format(
        self,
        on_channels_last: Callable[[], Any],
        on_channels_first: Callable[[], Any],
    ) -> Any:
        data_format = self.config['data_format']
        if data_format == 'channels_last':
            return on_channels_last()
        elif data_format == 'channels_first':
            return on_channels_first()
        else:
            raise NotImplementedError(f'Unsupported data_format {data_format}.')
