from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Sequence


@dataclass
class Dataset():
    ml_task: str
    input_shape: Sequence[int]
    output_shape: Sequence[int]
    train_data: Any
    validation_data: Any
    test_data: Any
    
