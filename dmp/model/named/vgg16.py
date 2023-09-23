from dataclasses import dataclass
from dataclasses import dataclass
from dmp.keras_interface.keras_utils import keras_kwcfg
from dmp.layer.avg_pool import AvgPool
from dmp.layer.dense import Dense
from dmp.layer.dense_conv import DenseConv
from dmp.layer.flatten import Flatten
from dmp.layer.global_average_pooling import GlobalAveragePooling
from dmp.layer.max_pool import MaxPool
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.layer_factory_model import LayerFactoryModel
from dmp.model.model_spec import ModelSpec
from dmp.model.network_info import NetworkInfo
from dmp.structure.batch_norm_block import BatchNormBlock
from dmp.structure.res_net_block import ResNetBlock
from dmp.structure.sequential_model import SequentialModel


@dataclass
class VGG16(ModelSpec):
    def make_network(self) -> NetworkInfo:
        conv_config = {
            "padding": "same",
            "use_bias": False,
            "kernel_constraint": keras_kwcfg("ParameterMask"),
        }

        # RETHINKING THE VALUE OF NETWORK PRUNING: https://arxiv.org/pdf/1810.05270.pdf
        # reference implementation: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/lottery-ticket/l1-norm-pruning/models/vgg.py
        # Original VGG: https://arxiv.org/pdf/1409.1556.pdf

        return LayerFactoryModel(
            SequentialModel(
                [
                    BatchNormBlock(DenseConv.make(64, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(64, [3, 3], [1, 1], conv_config)),
                    MaxPool.make([2, 2], [2, 2]),
                    BatchNormBlock(DenseConv.make(128, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(128, [3, 3], [1, 1], conv_config)),
                    MaxPool.make([2, 2], [2, 2]),
                    BatchNormBlock(DenseConv.make(256, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(256, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(256, [3, 3], [1, 1], conv_config)),
                    MaxPool.make([2, 2], [2, 2]),
                    BatchNormBlock(DenseConv.make(512, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(512, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(512, [3, 3], [1, 1], conv_config)),
                    MaxPool.make([2, 2], [2, 2]),
                    BatchNormBlock(DenseConv.make(512, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(512, [3, 3], [1, 1], conv_config)),
                    BatchNormBlock(DenseConv.make(512, [3, 3], [1, 1], conv_config)),
                    AvgPool.make([2, 2], [2, 2]),  # MaxPool in original paper
                    Flatten(),
                    # Dense.make(512),
                    # Dense.make(512),
                ]
            ),
            input=self.input,
            output=self.output,
        ).make_network()
