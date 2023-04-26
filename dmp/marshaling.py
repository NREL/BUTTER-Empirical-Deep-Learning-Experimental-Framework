from typing import Iterable, Type
from dmp.keras_interface.keras_utils import (
    register_custom_keras_type,
    register_custom_keras_types,
)
from dmp.common import marshal_type_key

from lmarshal.src.marshal import Marshal
from lmarshal.src.marshal_config import MarshalConfig

marshal_settings = {
    "type_key": marshal_type_key,
    "label_key": "label",
    "reference_prefix": "*",
    "escape_prefix": "\\",
    "flat_dict_key": ":",
    "enum_value_key": "value",
    "label_all": False,
    "label_referenced": True,
    "circular_references_only": True,
    "reference_strings": False,
}

marshal_config = MarshalConfig(**marshal_settings)

flat_marshal_settings = marshal_settings.copy()
flat_marshal_settings.update(
    {
        "circular_references_only": True,
    }
)

flat_marshal_config = MarshalConfig(**flat_marshal_settings)

marshal: Marshal = Marshal(marshal_config)


def register_types(
    target_types: Iterable[Type],
) -> None:
    marshal.register_types(target_types)


def register_type(
    target_type: Type,
    type_code=None,
    object_marshaler=None,
    demarshaling_factory=None,
    demarshaling_initializer=None,
) -> None:
    marshal.register_type(
        target_type,
        type_code,
        object_marshaler,
        demarshaling_factory,
        demarshaling_initializer,
    )


# register types below -----

import uuid

register_type(
    uuid.UUID,
    "UUID",
    lambda m, s: {m.marshal_key("value"): str(s)},
    lambda d, s: uuid.UUID(s[d.marshal_key("value")]),
    lambda d, s, r: r,
)

# Layers:
from dmp.layer.max_pool import MaxPool
from dmp.layer.avg_pool import AvgPool
from dmp.layer.global_average_pooling import GlobalAveragePooling
from dmp.layer.global_max_pooling import GlobalMaxPooling
from dmp.layer.dense import Dense
from dmp.layer.input import Input
from dmp.layer.add import Add
from dmp.layer.concatenate import Concatenate
from dmp.layer.identity import Identity
from dmp.layer.zeroize import Zeroize
from dmp.layer.dense_conv import DenseConv
from dmp.layer.separable_conv import SeparableConv
from dmp.layer.flatten import Flatten
from dmp.layer.op_layer import OpLayer
from dmp.layer.batch_normalization import BatchNormalization

# Layers:
register_types(
    [
        MaxPool,
        AvgPool,
        GlobalAveragePooling,
        GlobalMaxPooling,
        Dense,
        Input,
        Add,
        Concatenate,
        Identity,
        Zeroize,
        DenseConv,
        SeparableConv,
        Flatten,
        OpLayer,
        BatchNormalization,
    ]
)

# Tasks:
from dmp.postgres_interface.update_experiment_summary import UpdateExperimentSummary
from dmp.task.experiment.growth_experiment.growth_experiment import GrowthExperiment
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.experiment.pruning_experiment.iterative_pruning_experiment import (
    IterativePruningExperiment,
)

register_types(
    [
        UpdateExperimentSummary,
        TrainingExperiment,
        GrowthExperiment,
        IterativePruningExperiment,
    ]
)

# register summarization types
UpdateExperimentSummary.register_types(
    [
        TrainingExperiment,
        GrowthExperiment,
        IterativePruningExperiment,
    ]
)

# ModelSpec's:
from dmp.model.dense_by_size import DenseBySize
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
from dmp.model.fully_connected_network import FullyConnectedNetwork
from dmp.model.sequential_model import SequentialModel
from dmp.model.repeated_model import RepeatedModel

# LayerFactory's
from dmp.model.res_net_block import ResNetBlock

register_types(
    (
        DenseBySize,
        CNNStack,
        CNNStacker,
        FullyConnectedNetwork,
        SequentialModel,
        RepeatedModel,
        ResNetBlock,
    )
)

# stopping methods, growth triggers, callbacks
from dmp.task.experiment.growth_experiment.growth_trigger.proportional_stopping import (
    ProportionalStopping,
)
from dmp.task.experiment.pruning_experiment.weight_mask import WeightMask

register_custom_keras_types(
    {
        "ProportionalStopping": ProportionalStopping,
        "WeightMask": WeightMask,
    }
)

# scaling methods
from dmp.task.experiment.growth_experiment.scaling_method.width_scaler import (
    WidthScaler,
)

register_types((WidthScaler,))

# transfer methods
from dmp.task.experiment.growth_experiment.transfer_method.overlay_transfer import (
    OverlayTransfer,
)

register_types((OverlayTransfer,))

# pruning methods:
from dmp.task.experiment.pruning_experiment.pruning_method.magnitude_pruner import (
    MagnitudePruner,
)

register_types((MagnitudePruner,))

# Other types:
from dmp.task.experiment.training_experiment.experiment_record_settings import (
    ExperimentRecordSettings,
)
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.recorder.test_set_history_recorder import (
    TestSetHistoryRecorder,
)
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask
from dmp.model.network_info import NetworkInfo

register_types(
    (
        ExperimentRecordSettings,
        ExperimentResultRecord,
        TestSetHistoryRecorder,
        DatasetSpec,
        MLTask,
        NetworkInfo,
    )
)
