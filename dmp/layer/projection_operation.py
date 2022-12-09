from typing import Any, Dict, Sequence, Tuple, Callable, TypeVar, List, Union
from dmp.layer.convolutional_layer import ConvolutionalLayer
from dmp.layer.layer import network_module_types

class ProjectionOperation(ConvolutionalLayer):
    pass


network_module_types.append(ProjectionOperation)


