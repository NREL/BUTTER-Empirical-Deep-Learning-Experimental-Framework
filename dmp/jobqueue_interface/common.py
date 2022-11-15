

from dmp.structure.n_add import NAdd
from dmp.structure.n_dense import NDense
from dmp.structure.n_input import NInput
from dmp.structure.n_conv import *
from dmp.structure.n_cell import *
from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment
from lmarshal import MarshalConfig, Marshal

jobqueue_marshal: Marshal = Marshal(MarshalConfig(
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
jobqueue_marshal.register_type(GrowthExperiment)


# register NetworkModule here
jobqueue_marshal.register_type(NInput)
jobqueue_marshal.register_type(NDense)
jobqueue_marshal.register_type(NAdd)

# CNN Modules
jobqueue_marshal.register_type(NBasicCNN)
jobqueue_marshal.register_type(NCNNInput)
jobqueue_marshal.register_type(NConv)
jobqueue_marshal.register_type(NSepConv)
jobqueue_marshal.register_type(NMaxPool)
jobqueue_marshal.register_type(NGlobalPool)
jobqueue_marshal.register_type(NIdentity)
jobqueue_marshal.register_type(NZeroize)

#CNN Cell modules
jobqueue_marshal.register_type(NBasicCell)
jobqueue_marshal.register_type(NConvStem)
jobqueue_marshal.register_type(NCell)
jobqueue_marshal.register_type(NDownsample)
jobqueue_marshal.register_type(NFinalClassifier)

