from typing import Iterable, Optional, Type
from lmarshal import MarshalConfig, Marshal
from dmp.common import marshal_type_key
from lmarshal.src.marshal_types import DemarshalingFactory, DemarshalingInitializer, RawObjectMarshaler, TypeCode

jobqueue_marshal: Marshal = Marshal(
    MarshalConfig(type_key=marshal_type_key,
                  label_key='label',
                  reference_prefix='*',
                  escape_prefix='\\',
                  flat_dict_key=':',
                  enum_value_key='value',
                  label_all=False,
                  label_referenced=True,
                  circular_references_only=False,
                  reference_strings=False))

# from dmp.task.task import task_types

# jobqueue_marshal.register_types(task_types)

# from dmp.layer.layer import layer_types

# jobqueue_marshal.register_types(layer_types)


# def register_types(
#     target_types: Iterable[Type],
# ) -> None:
#     jobqueue_marshal.register_types(target_types)


# def register_type(
#     target_type: Type,
#     type_code: Optional[TypeCode] = None,
#     object_marshaler: Optional[ObjectMarshaler] = None,
#     demarshaling_factory: Optional[DemarshalingFactory] = None,
#     demarshaling_initializer: Optional[DemarshalingInitializer] = None,
# ) -> None:
#     jobqueue_marshal.register_type(
#         target_type,
#         type_code,
#         object_marshaler,
#         demarshaling_factory,
#         demarshaling_initializer,
#     )

import dmp.jobqueue_interface.registry





