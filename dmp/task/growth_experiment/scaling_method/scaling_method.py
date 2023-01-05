
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from dmp.layer.layer import Layer

from dmp.model.model_info import ModelInfo
from dmp.task.growth_experiment.layer_growth_info import LayerGrowthInfo


class ScalingMethod(ABC):
    
    @abstractmethod
    def scale(
        self,
        target: Layer, 
        scale_factor: float,
    ) -> Tuple[Layer, Dict[Layer, Layer]]:
        pass