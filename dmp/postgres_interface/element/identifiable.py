from abc import ABC, abstractproperty
from psycopg.sql import Identifier

class Identifiable(ABC):

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def identifier(self) -> Identifier:
        return Identifier(self.name)