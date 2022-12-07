from dataclasses import dataclass, field
from typing import Optional, Sequence, Any, Tuple, Dict
from dmp.layer.layer import Layer

@dataclass
class NetworkInfo():
    structure: Layer
    description: Dict[str, Any]
    num_free_parameters : int = -1
    
