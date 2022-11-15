from dataclasses import dataclass
from typing import Optional, Union
from dmp.structure.n_neuron_layer import NNeuronLayer


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NDense(NNeuronLayer):

    @property
    def num_free_parameters_in_module(self) -> int:
        return (sum((i.size for i in self.inputs)) + 1) * self.size