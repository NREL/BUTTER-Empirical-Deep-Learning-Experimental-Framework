from dmp.marshal_registry import register_type, register_types
import uuid
from dmp.dataset.ml_task import MLTask
from dmp.layer.layer import LayerFactory
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
import dmp.task
import dmp.layer

from dmp.model.dense_by_size import DenseBySize
from dmp.task.growth_experiment.growth_trigger.proportional_stopping import ProportionalStopping

from dmp.task.task_result_record import TaskResultRecord
from dmp.task.growth_experiment.transfer_method.overlay_transfer import OverlayTransfer
from dmp.task.recorder.test_set_history_recorder import TestSetHistoryRecorder
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask

from dmp.keras_interface.keras_utils import register_custom_keras_type

# register types below -----

register_type(
    uuid.UUID,
    'UUID',
    lambda m, s: {m.marshal_key('value'): str(s)},
    lambda d, s: uuid.UUID(s[d.marshal_key('value')]),
    lambda d, s, r: r,
)

# ModelSpec's:
register_types((
    DenseBySize,
    CNNStack,
    CNNStacker,
))

# stopping methods and growth triggers
register_custom_keras_type('ProportionalStopping', ProportionalStopping)

# Other types:
register_types((
    TaskResultRecord,
    OverlayTransfer,
    TestSetHistoryRecorder,
    DatasetSpec,
    MLTask,
))

