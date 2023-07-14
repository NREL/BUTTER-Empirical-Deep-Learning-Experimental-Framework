from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional, Union

import tensorflow.keras as keras
from dmp.model.keras_network_info import KerasNetworkInfo
from dmp.model.network_info import NetworkInfo


@dataclass
class ModelInfo:
    """
    Holds information about a model that has been instantiated as a keras model.
    """

    network: NetworkInfo  # information about the structure and configuration of the network layers
    keras_network: KerasNetworkInfo  # cooresponding keras model and associated mapping
    keras_model: keras.Model  # the keras model object itself
