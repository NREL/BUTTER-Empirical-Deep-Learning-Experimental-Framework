from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from lmarshal.src.marshal import Marshal
from lmarshal.src.marshal_config import MarshalConfig


task_marshal = Marshal(MarshalConfig(
        type_key='%',
        label_key='&',
        reference_prefix='*',
        escape_prefix='!',
        flat_dict_key=':',
        label_all=False,
        label_referenced=False,
        circular_references_only=False,
        reference_strings=False))


task_marshal.register_type(AspectTestTask)

