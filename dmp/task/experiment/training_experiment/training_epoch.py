from dataclasses import dataclass
from typing import Any, Dict, Optional


"""
from dmp.marshaling import marshal

my_epoch = TrainingEpoch(1, 2, 3)
marshaled = marshal.marshal(my_epoch)

{
    'epoch' : 1,
    'model_number' : 2,
    'model_epoch' : 3,
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
    model_number: int
    model_epoch: int

    def count_new_model(self) -> None:
        self.model_number += 1
        self.model_epoch = 0

    def count_new_epoch(self) -> None:
        self.epoch += 1
        self.model_epoch += 1
