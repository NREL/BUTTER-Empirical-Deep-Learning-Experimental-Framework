from typing import Optional, Type

from lmarshal.demarshaler import Demarshaler
from lmarshal.marshal_config import MarshalConfig
from lmarshal.marshaler import Marshaler
from lmarshal.types import ObjectDemarshaler, ObjectMarshaler, TypeCode


class Marshal:
    """

    """

    def __init__(
            self,
            config: MarshalConfig,
    ) -> None:
        self._config: MarshalConfig = config

    def register_type(
            self,
            target_type: Type,
            marshaler: Optional[ObjectMarshaler] = None,
            demarshaler: Optional[ObjectDemarshaler] = None,
            type_code: Optional[TypeCode] = None,
    ) -> None:
        return self._config.register_type(target_type, marshaler, demarshaler, type_code)

    def marshal(self, target: any) -> any:
        return Marshaler(self._config)(target)

    def demarshal(self, target: any) -> any:
        return Demarshaler(self._config)(target)
