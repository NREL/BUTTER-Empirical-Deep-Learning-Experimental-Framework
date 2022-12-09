from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, Callable


class LayerFactory(ABC):

    @abstractmethod
    def make_layer(self, inputs: List['Layer']) -> 'Layer':
        pass


from .layer import Layer
