from dataclasses import dataclass
from dataclasses import dataclass
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.layer.dense import Dense
from dmp.layer.dense_conv import DenseConv
from dmp.layer.flatten import Flatten
from dmp.layer.global_average_pooling import GlobalAveragePooling
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.layer_factory_model import LayerFactoryModel
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo
from dmp.structure.batch_norm_block import BatchNormBlock
from dmp.structure.res_net_block import ResNetBlock
from dmp.structure.sequential_model import SequentialModel


@dataclass
class Resnet20(ModelSpec):
    def make_network(self) -> NetworkInfo:
        return LayerFactoryModel(
            SequentialModel(
                [
                    BatchNormBlock(
                        DenseConv.make(
                            16,
                            [7, 7],
                            [2, 2],
                            {
                                "padding": "same",
                                "use_bias": False,
                                "kernel_constraint": keras_kwcfg("ParameterMask"),
                            },
                        )
                    ),
                    ResNetBlock(16, 1),
                    ResNetBlock(16, 1),
                    ResNetBlock(16, 1),
                    ResNetBlock(32, 2),
                    ResNetBlock(32, 1),
                    ResNetBlock(32, 1),
                    ResNetBlock(64, 2),
                    ResNetBlock(64, 1),
                    ResNetBlock(64, 1),
                    GlobalAveragePooling(),
                    Flatten(),
                ]
            ),
            input=self.input,
            output=self.output,
        ).make_network()
