from lmarshal.marshal_config import MarshalConfig


class CommonMarshaler:
    __slots__ = ['_config']

    def __init__(self, config: MarshalConfig) -> None:
        self._config = config

    def _make_label(self, index: int):
        return hex(index)[2:]

    def _escape_string(self, source: str) -> str:
        return self._config.escape_prefix + source

    def _unescape_string(self, source: str) -> str:
        return source[len(self._config.escape_prefix):]
