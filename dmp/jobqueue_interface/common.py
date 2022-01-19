


from dmp.experiment.structure.n_add import NAdd
from dmp.experiment.structure.n_dense import NDense
from dmp.experiment.structure.n_input import NInput
from lmarshal import Marshal, MarshalConfig
from dmp.experiment.task.aspect_test_task import AspectTestTask
from lmarshal.src.demarshaler import Demarshaler

jobqueue_marshal =  Marshal(MarshalConfig(
        type_key='',
        label_key='&',
        reference_prefix='*',
        escape_prefix='!',
        flat_dict_key=':',
        label_all=False,
        label_referenced=True,
        circular_references_only=False,
        reference_strings=False))


# register Task here
jobqueue_marshal.register_type(AspectTestTask)


# register NetworkModule here
jobqueue_marshal.register_type(NInput)
jobqueue_marshal.register_type(NDense)
jobqueue_marshal.register_type(NAdd)

