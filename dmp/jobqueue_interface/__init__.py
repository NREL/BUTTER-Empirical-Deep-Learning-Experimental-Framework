from typing import Tuple, Type
from lmarshal import MarshalConfig, Marshal
import dmp.structure
import dmp.task

jobqueue_marshal: Marshal = Marshal(
    MarshalConfig(type_key='',
                  label_key='&',
                  reference_prefix='*',
                  escape_prefix='!',
                  flat_dict_key=':',
                  label_all=False,
                  label_referenced=True,
                  circular_references_only=False,
                  reference_strings=False))

jobqueue_marshal.register_types(dmp.task.task_types)

jobqueue_marshal.register_types(dmp.structure.network_module_types)
