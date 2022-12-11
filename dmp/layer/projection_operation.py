from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import AConvolutionalLayer
from dmp.layer.layer import network_module_types

class ProjectionOperation(AConvolutionalLayer):
    pass


network_module_types.append(ProjectionOperation)


