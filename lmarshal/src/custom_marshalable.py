from abc import ABC, abstractmethod
from .marshaler import Marshaler
from .demarshaler import Demarshaler


class CustomMarshalable(ABC):

    @abstractmethod
    def marshal(self, marshaler: Marshaler) -> dict:
        pass

    @abstractmethod
    def demarshal(self, demarshaler: Demarshaler, source: dict) -> None:
        pass