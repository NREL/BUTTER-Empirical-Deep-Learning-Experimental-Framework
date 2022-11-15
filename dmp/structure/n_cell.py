from dataclasses import dataclass, field
from typing import List, Optional

from dmp.structure.network_module import NetworkModule
from dmp.structure.n_neuron_layer import NNeuronLayer
from dmp.structure.n_dense import NDense
from dmp.structure.n_add import NAdd
from dmp.structure.n_conv import *


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConvolutionalCell(NNeuronLayer):
    filters: int = 16


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NConvStem(NConvolutionalCell):
    batch_norm: str = 'none'
    input_channels: int = 3

    @property
    def num_free_parameters_in_module(self) -> int:
        return 9 * self.filters * self.input_channels


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NCell(NConvolutionalCell):
    batch_norm: str = 'none'
    operations: List[str] = field(default_factory=list)
    nodes: int = 2
    cell_type: str = 'graph'

    @property
    def num_free_parameters_in_module(self) -> int:
        if self.cell_type == 'parallelconcat':
            filters = [self.filters // self.nodes for _ in range(self.nodes)]
            for i in range(self.filters % self.nodes):
                filters[i] += 1
        else:
            filters = [self.filters for _ in range(self.nodes)]
        params = 0
        in_channels = self.convolutional_inputs[0].filters
        params_dict = {
            'conv3x3': 9,
            'conv5x5': 25,
            'sepconv3x3': 6,
            'sepconv5x5': 10,
            'conv1x1': 1,
            'maxpool3x3': 0,
            'avgpool3x3': 0,
            'identity': 0,
            'zeroize': 0,
            'projection': 0
        }
        for i in range(self.nodes):
            num_channels = filters[i]
            ops = self.operations[i]
            for j in range(len(ops)):
                op = ops[j]
                if j > 0:
                    params += params_dict[op] * num_channels**2
                else:
                    params += params_dict[op] * num_channels * in_channels
        return params


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NDownsample(NConvolutionalCell):

    @property
    def num_free_parameters_in_module(self) -> int:
        params = 1
        for i in self.inputs:
            params *= i.filters
        params *= self.filters
        return params


@dataclass(frozen=False, eq=False, unsafe_hash=False)
class NFinalClassifier(NetworkModule):
    classes: int = 10
    activation: str = 'softmax'
    kernel_regularizer: Optional[dict] = None
    bias_regularizer: Optional[dict] = None
    activity_regularizer: Optional[dict] = None

    @property
    def num_free_parameters_in_module(self) -> int:
        params = 1
        for i in self.inputs:
            params *= i.filters
        params *= self.classes
        return params


########################################################################################
#--------------------------------------------------------------------------------------#
#                        Cell Generators
#--------------------------------------------------------------------------------------#
########################################################################################


def generate_conv_stem(
    inputs,
    filters=16,
    batch_norm='none',
    activation='relu',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    module = NConv(
        label=0,
        inputs=[
            inputs,
        ],
        filters=filters,
        kernel_size=3,
        stride=1,
        padding='same',
        batch_norm=batch_norm,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
    )
    return module


def generate_downsample(
    inputs,
    filters=16,
    batch_norm='none',
    activation='relu',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    module = NMaxPool(
        label=0,
        inputs=[
            inputs,
        ],
        filters=filters,
        kernel_size=2,
        stride=2,
        padding='same',
        activation=activation,
    )
    module = NConv(
        label=0,
        inputs=[
            module,
        ],
        filters=filters,
        kernel_size=1,
        stride=1,
        padding='same',
        batch_norm=batch_norm,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
    )
    return module


def generate_final_classifier(
    inputs,
    classes=10,
    activation='softmax',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    module = NGlobalPool(
        label=0,
        inputs=[
            inputs,
        ],
    )
    module = NDense(
        label=0,
        inputs=[
            module,
        ],
        shape=[
            classes,
        ],
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
    )
    return module


########################################################################################
#--------------------------------------------------------------------------------------#
#                        Generic Cell Generators
#--------------------------------------------------------------------------------------#
########################################################################################


def generate_module(
    inp,
    op,
    filters,
    batch_norm,
    activation,
    kernel_regularizer,
    bias_regularizer,
    activity_regularizer,
):
    module = None
    if op == 'conv3x3':
        module = NConv(
            label=0,
            inputs=[
                inp,
            ],
            filters=filters,
            kernel_size=3,
            stride=1,
            padding='same',
            batch_norm=batch_norm,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
    elif op == 'conv5x5':
        module = NConv(
            label=0,
            inputs=[
                inp,
            ],
            filters=filters,
            kernel_size=5,
            stride=1,
            padding='same',
            batch_norm=batch_norm,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
    elif op == 'conv1x1':
        module = NConv(
            label=0,
            inputs=[
                inp,
            ],
            filters=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            batch_norm=batch_norm,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
    elif op == 'sepconv3x3':
        module = NSepConv(
            label=0,
            inputs=[
                inp,
            ],
            filters=filters,
            kernel_size=3,
            stride=1,
            padding='same',
            batch_norm=batch_norm,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
    elif op == 'sepconv5x5':
        module = NSepConv(
            label=0,
            inputs=[
                inp,
            ],
            filters=filters,
            kernel_size=5,
            stride=1,
            padding='same',
            batch_norm=batch_norm,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
    elif op == 'maxpool3x3':
        module = NMaxPool(
            label=0,
            inputs=[
                inp,
            ],
            filters=filters,
            kernel_size=3,
            stride=1,
            padding='same',
            activation=activation,
        )
    elif op == 'identity':
        module = NIdentity(
            label=0,
            inputs=[
                inp,
            ],
        )
    elif op == 'zeroize':
        module = NZeroize(
            label=0,
            inputs=[
                inp,
            ],
        )
    else:
        raise ValueError(f'Unknown operation {op}')

    return module


def generate_graph_cell(
    inputs,
    nodes,
    operations,
    filters=16,
    batch_norm='none',
    activation='relu',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    # Graph cell connects node i to all nodes j where j>i and sums at every node
    # nodes is the number of nodes where operations are summed in the cell with node 1 being the input tensor
    # operations is a list of lists operations corresponding to each node
    assert nodes >= 2  # 'nodes must be greater than or equal to 2'
    assert len(
        operations
    ) == nodes - 1  # 'operations must be a list of lists of operations corresponding to each node'
    node_list = [inputs] + [None] * (nodes - 1)
    inds = [0 for _ in range(nodes - 1)]
    storage = {}
    for i in range(1, nodes):
        ops = operations[i - 1]
        inp = node_list[i - 1]
        storage[i - 1] = []
        for j in range(len(ops)):
            op = ops[j]
            module = generate_module(
                inp,
                op,
                filters,
                batch_norm,
                activation,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
            )
            storage[i - 1].append(module)
        ins = [storage[k][inds[k]] for k in range(i)]
        if len(ins) > 1:
            node = NAdd(
                label=0,
                inputs=ins,  # all the operations at node i
            )
        else:
            node = ins[0]
        for k in range(i):
            inds[k] += 1
        node_list[i] = node
    return node_list[-1]


def generate_parallel_concat_cell(
    inputs,
    nodes,
    operations,
    filters=16,
    batch_norm='none',
    activation='relu',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    # Nodes is the number of parallel tracks
    # Operations is a list of lists corresponding to the operations at each node
    assert len(operations) == nodes
    channel_list = [filters // nodes for _ in range(nodes)]
    for i in range(filters % nodes):
        channel_list[i] += 1
    by_node = [len(operations[i]) for i in range(nodes)]
    tracks = [None for _ in range(nodes)]
    module = None
    for i in range(nodes):
        filters = channel_list[i]
        ops = operations[i]
        for j in range(by_node[i]):
            inp = inputs if j == 0 else module
            op = ops[j]
            module = generate_module(
                inp,
                op,
                filters,
                batch_norm,
                activation,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
            )
        tracks[i] = module
    if len(tracks) > 1:
        module = NConcat(
            label=0,
            inputs=tracks,
        )
    else:
        module = tracks[0]

    return module


def generate_parallel_add_cell(
    inputs,
    nodes,
    operations,
    filters=16,
    batch_norm='none',
    activation='relu',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    # Nodes is the number of parallel tracks
    # Operations is a list of lists corresponding to the operations at each node
    assert len(operations) == nodes
    by_node = [len(operations[i]) for i in range(nodes)]
    tracks = [None for _ in range(nodes)]
    module = None
    for i in range(nodes):
        ops = operations[i]
        for j in range(by_node[i]):
            inp = inputs if j == 0 else module
            op = ops[j]
            module = generate_module(
                inp,
                op,
                filters,
                batch_norm,
                activation,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
            )
        tracks[i] = module
    if len(tracks) > 1:
        module = NAdd(
            label=0,
            inputs=tracks,
        )
    else:
        module = tracks[0]

    return module


def generate_generic_cell(
    type,
    inputs,
    nodes,
    operations,
    filters=16,
    batch_norm='none',
    activation='relu',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
):
    args = {
        'inputs': inputs,
        'nodes': nodes,
        'operations': operations,
        'filters': filters,
        'batch_norm': batch_norm,
        'activation': activation,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'activity_regularizer': activity_regularizer,
    }
    if type == 'graph':
        return generate_graph_cell(**args)
    elif type == 'parallelconcat':
        return generate_parallel_concat_cell(**args)
    elif type == 'paralleladd':
        return generate_parallel_add_cell(**args)
    else:
        raise ValueError(f'Invalid cell type: {type}')