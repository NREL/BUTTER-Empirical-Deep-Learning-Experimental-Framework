from dmp.marshal_registry import register_type, register_types
import uuid
from dmp.dataset.ml_task import MLTask
from dmp.layer.layer import LayerFactory
from dmp.model.cnn.cnn_stack import CNNStack
from dmp.model.cnn.cnn_stacker import CNNStacker
import dmp.task
import dmp.layer

from dmp.model.dense_by_size import DenseBySize
from dmp.task.task_result_record import TaskResultRecord
from dmp.task.growth_experiment.growth_method.overlay_growth_method import OverlayGrowthMethod
from dmp.task.training_experiment.test_set_history_recorder import TestSetHistoryRecorder
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask

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

# Other types:
register_types((
    TaskResultRecord,
    OverlayGrowthMethod,
    TestSetHistoryRecorder,
    DatasetSpec,
    MLTask,
))
