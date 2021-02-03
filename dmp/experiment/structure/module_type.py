from enum import IntEnum, auto


class ModuleType(IntEnum):
    FULLY_CONNECTED = auto()
    INPUT = auto()
    ADD = auto()
