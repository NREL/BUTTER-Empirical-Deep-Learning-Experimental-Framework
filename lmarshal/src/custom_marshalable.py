from abc import ABC, abstractmethod
from .marshaler import Marshaler
from .demarshaler import Demarshaler


class CustomMarshalable(ABC):

    @abstractmethod
    def marshal(self) -> dict:
        pass

    @abstractmethod
    def demarshal(self, source: dict) -> None:
        pass