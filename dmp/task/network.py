from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional
import tensorflow.keras as keras
import tensorflow

from dmp.layer.layer import Layer
from dmp.layer.visitor.keras_interface.layer_to_keras import KerasLayer


@dataclass
class Network():
    network_structure: Layer
    layer_shapes: Dict[Layer, Tuple]
    widths: List[int]
    num_free_parameters: int
    output_activation: str
    layer_to_keras_map: Dict[Layer, Tuple[KerasLayer, tensorflow.Tensor]]
    keras_model: keras.Model

    def compile_model(self, optimizer: Dict, loss: Any) -> None:
        run_metrics = [
            'accuracy',
            keras.metrics.CosineSimilarity(),
            keras.metrics.Hinge(),
            keras.metrics.KLDivergence(),
            keras.metrics.MeanAbsoluteError(),
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanSquaredLogarithmicError(),
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.SquaredHinge(),
        ]

        run_optimizer = keras.optimizers.get(optimizer)
        self.keras_model.compile(
            loss=loss,
            optimizer=run_optimizer,  # type: ignore
            metrics=run_metrics,
            run_eagerly=False,
        )