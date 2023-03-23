from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional, Union

import tensorflow.keras as keras
from dmp.model.keras_network_info import KerasNetworkInfo
from dmp.model.network_info import NetworkInfo


@dataclass
class ModelInfo():
    network: NetworkInfo
    keras_network: KerasNetworkInfo
    keras_model: keras.Model
