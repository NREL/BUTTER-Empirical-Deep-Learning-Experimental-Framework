from dataclasses import dataclass
from typing import Any, Dict, Optional


"""
from dmp.marshaling import marshal

my_epoch = TrainingEpoch(1, 2, 3)
marshaled = marshal.marshal(my_epoch)

{
    'epoch' : 1,
    'fit_number' : 2,
    'fit_epoch' : 3,
    'type' : 'TrainingEpoch',
}

import json
serialized = json.dumps(marshaled)

deserialized = json.loads(serialized)
recovered_epoch = marshal.demarshal(deserialized)



"""


@dataclass(order=True)
class TrainingEpoch:
    epoch: int
    fit_number: int
    fit_epoch: int
    marker: int = 0  # 0: regular, 1: final (best)
    sequence_number: Optional[int] = None

    def count_new_model(self) -> None:
        self.fit_number += 1
        self.fit_epoch = 0
        self.marker = 0
        self.sequence_number = None

    def count_new_epoch(self, delta: int = 1) -> None:
        self.epoch += delta
        self.fit_epoch += delta
        self.marker = 0
        self.sequence_number = None
