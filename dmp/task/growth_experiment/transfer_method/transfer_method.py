
from abc import ABC, abstractmethod
from typing import Dict, List
from dmp.layer.layer import Layer

from dmp.model.model_info import ModelInfo
from dmp.task.growth_experiment.layer_growth_info import LayerGrowthInfo


class TransferMethod(ABC):
    '''
    Transfers parameters from one network's layers into another likely different
    network.
    For example, if you have a small trained network and you want to transfer
    its weights into a larger network, a transfer method can do that.
    '''
    
    @abstractmethod
    def transfer(
        self,
        growth_map: List[LayerGrowthInfo] # maps src layers to dst layers
    ) -> None:
        pass