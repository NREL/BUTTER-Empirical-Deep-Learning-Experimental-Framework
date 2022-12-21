
from abc import ABC, abstractmethod
from typing import Dict
from dmp.layer.layer import Layer

from dmp.model.model_info import ModelInfo
from dmp.task.growth_experiment.layer_growth_info import LayerGrowthInfo


class GrowthMethod(ABC):
    
    @abstractmethod
    def grow(
        self,
        src: ModelInfo,  # 'parent' / 'previous' network
        dest: ModelInfo,  # 'child' / 'next' network
        growth_map: Dict[Layer, LayerGrowthInfo]
    ) -> None:
        pass