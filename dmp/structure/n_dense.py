from dataclasses import dataclass, field
from typing import Optional, Union, List, Sequence
from dmp.structure.n_neuron_layer import NNeuronLayer


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NDense(NNeuronLayer):
    shape: List[int] = field(default_factory=list)

    @property
    def num_free_parameters_in_module(self) -> int:
        return self.output_size * \
            (sum((i.output_size for i in self.inputs)) +\
                 (1 if self.use_bias else 0))

    @property
    def output_shape(self) -> Sequence[int]:
        return self.shape

    @property
    def dimension(self) -> int:
        return len(self.shape)