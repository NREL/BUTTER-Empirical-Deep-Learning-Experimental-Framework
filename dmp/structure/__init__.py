from typing import Tuple, Type
from dmp.structure.n_add import NAdd
from dmp.structure.n_dense import NDense
from dmp.structure.n_input import NInput
from dmp.structure.n_conv import *
from dmp.structure.n_cell import *

# register NetworkModule subclasses here
network_module_types: Tuple[Type, ...] = (
    NInput,
    NDense,
    NAdd,

    # CNN Modules
    NBasicCNN,
    # NCNNInput,
    NConv,
    NSepConv,
    NMaxPool,
    NGlobalPool,
    NIdentity,
    NZeroize,

    # CNN Cell modules
    NConvolutionalCell,
    NConvStem,
    NDownsample,
    NFinalClassifier,
    # NCell,
)
