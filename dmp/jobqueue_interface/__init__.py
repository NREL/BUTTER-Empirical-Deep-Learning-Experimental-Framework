from typing import Tuple, Type
from lmarshal import MarshalConfig, Marshal
import dmp.layer.layer
import dmp.task

tensorflow_type_key :str = 'class_name'
tensorflow_config_key:str = 'config'
keras_type_key: str = 'type'
marshal_type_key:str = 'class'


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

jobqueue_marshal.register_types(dmp.task.task_types)

jobqueue_marshal.register_types(dmp.layer.layer.network_module_types)