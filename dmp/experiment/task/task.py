from abc import abstractmethod
from dataclasses import dataclass

from numpy.testing._private.utils import integer_repr
from dmp.experiment.structure.network_module import NetworkModule

class Task:

    @property
    @abstractmethod
    def type(self) -> TaskType:


@dataclass
class TrainNeuralNetWithKerasTask(Task):
    seed: int
    network: NetworkModule
    dataset: str


class DenseNeuralNetFromParameters(Task):
    seed: int
    width: int
    dataset: str
