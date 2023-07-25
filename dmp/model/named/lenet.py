from dmp.layer.dense import Dense
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.fully_connected_network import FullyConnectedNetwork


class Lenet(CNNStack):
    def __init__(self):
        super().__init__(
            input=None,
            output=None,
            num_stacks=2,
            cells_per_stack=1,
            stem="conv_5x5_1x1_same",
            downsample="max_pool_2x2_2x2_valid",
            cell="conv_5x5_1x1_valid",
            final=FullyConnectedNetwork(
                input=None,
                output=None,
                widths=[120, 84],
                residual_mode="none",
                flatten_input=True,
                inner=Dense.make(-1, {}),
            ),
            stem_width=6,
            stack_width_scale_factor=16.0 / 6.0,
            downsample_width_scale_factor=1.0,
            cell_width_scale_factor=1.0,
        )
