from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List, Union
from dmp.postgres_interface.element.identifiable import Identifiable


class Table(Identifiable):
    _name: str

    def __init__(self, name:str) -> None:
        super().__init__()
        self._name = name


    
