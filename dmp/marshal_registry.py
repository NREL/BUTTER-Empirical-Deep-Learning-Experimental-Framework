from typing import Iterable, Optional, Type

def register_types(
    target_types: Iterable[Type],
) -> None:
    from dmp.jobqueue_interface import jobqueue_marshal
    jobqueue_marshal.register_types(target_types)


def register_type(
    target_type: Type,
    type_code = None,
    object_marshaler = None,
    demarshaling_factory = None,
    demarshaling_initializer = None,
) -> None:
    from dmp.jobqueue_interface import jobqueue_marshal
    jobqueue_marshal.register_type(
        target_type,
        type_code,
        object_marshaler,
        demarshaling_factory,
        demarshaling_initializer,
    )