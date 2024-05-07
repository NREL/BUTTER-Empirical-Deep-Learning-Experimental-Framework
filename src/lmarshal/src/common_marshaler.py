from .marshal_config import MarshalConfig


class CommonMarshaler:
    __slots__ = ['_config']

    def __init__(self, config: MarshalConfig) -> None:
        self._config = config

    def marshal_key(self, source: str) -> str:
        if source in self._config.control_key_set or \
            source.startswith(self._config.escape_prefix):
            # TODO: optionally allow referenced strings as keys?
            source = self._escape_string(source)
        return source

    def demarshal_key(self, source: str) -> str:
        if source.startswith(self._config.escape_prefix):
            source = self._unescape_string(source)
        return source
        
    def _make_label(self, index: int):
        return hex(index)[2:]

    def _escape_string(self, source: str) -> str:
        return self._config.escape_prefix + source

    def _unescape_string(self, source: str) -> str:
        return source[len(self._config.escape_prefix):]
