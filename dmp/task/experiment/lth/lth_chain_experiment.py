from dataclasses import dataclass
from operator import index
import random
from typing import Any, Dict, Iterable, Optional, Set, Type
from uuid import UUID
from jobqueue.job import Job
import numpy
import tensorflow
from dmp import common
from dmp.common import KerasConfig
from dmp.model.network_info import NetworkInfo
from dmp.parquet_util import make_dataframe_from_dict
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder
from dmp.task.experiment.training_experiment import (
    training_experiment_keys,
    training_experiment_summarizer,
)
from dmp.task.experiment.training_experiment.experiment_record_settings import (
    ExperimentRecordSettings,
)
from dmp.task.experiment.training_experiment.model_saving_callback import (
    ModelSavingCallback,
)
from dmp.task.experiment.training_experiment.resume_config import ResumeConfig
from dmp.task.experiment.training_experiment.training_experiment import TrainingExperiment
import tensorflow.keras as keras

from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_config
from dmp.keras_interface.layer_to_keras import make_keras_model_from_network
from dmp.layer import *
from dmp.task.experiment.recorder.timestamp_recorder import TimestampRecorder
from dmp.task.experiment.recorder.test_set_history_recorder import (
    TestSetHistoryRecorder,
)
from dmp.task.experiment.recorder.zero_epoch_recorder import ZeroEpochRecorder
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
from dmp.task.experiment.experiment_task import ExperimentTask
from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.model.model_info import ModelInfo

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec

from dmp.worker import Worker


'''
Chain goes: 
    + Regular training run
        + savepoints at possible restart points, and then along the training trajectory
        + for each (early?) savepoint:
            + for [.8^(2) = .64 (36%), .8 (20%), .8^(1/2)~=.894 (10.6%), .8^(1/4) ~= .945 (5.4%)] pruning per IMP iteration to target of <3.5% LeNet (16 iters), 3.5% ResNet (16 iters), 0.6% (24 iters) VGG:
                + do IMP with savepoints at end of each training cycle (before pruning), maybe along the way too
                + same as above, but with a different data order

+ save file custom paths
    + edit DB
    + edit table class
    + edit save procedure
    + add configuration to SaveMode
    + add support to experiments
        + {experiment group}/{model}/{stage (full network, IMP_pct_0, IMP_pct_1)}/{model number}/{root run id}/{model epochs}/{run id}
+ LTH chain experiment
    + implement structure
    + implement ID passing / parent / base ID
    + 
'''

@dataclass
class LTHChainExperiment(TrainingExperiment):
    
    pass