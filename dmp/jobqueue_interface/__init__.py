from lmarshal import MarshalConfig, Marshal
from dmp.common import marshal_type_key

jobqueue_marshal: Marshal = Marshal(
    MarshalConfig(type_key=marshal_type_key,
                  label_key='label',
                  reference_prefix='*',
                  escape_prefix='\\',
                  flat_dict_key=':',
                  label_all=False,
                  label_referenced=True,
                  circular_references_only=False,
                  reference_strings=False))

from dmp.task.task import task_types
jobqueue_marshal.register_types(task_types)

from dmp.layer.layer import layer_types
jobqueue_marshal.register_types(layer_types)