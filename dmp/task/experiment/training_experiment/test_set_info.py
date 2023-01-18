from dataclasses import dataclass
from typing import Any

@dataclass
class TestSetInfo():
    history_key: str
    test_data: Any
    test_targets: Any = None
    sample_weights: Any = None

