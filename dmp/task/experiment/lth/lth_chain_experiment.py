from dataclasses import dataclass
import itertools
from operator import index
import random
from typing import Any, Dict, Iterable, Optional, Set, Type
from uuid import UUID
import uuid
from jobqueue.job import Job
import numpy
import tensorflow
from dmp import common
from dmp.common import KerasConfig
from dmp.model.network_info import NetworkInfo
from dmp.parquet_util import make_dataframe_from_dict
from dmp.task.experiment.delegating_experiment import DelegatingExperiment
from dmp.task.experiment.experiment_summary_record import ExperimentSummaryRecord
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.pruning_experiment.lth_seed_change_experiment import LTHSeedChangeExperiment
from dmp.task.experiment.pruning_experiment.pruning_iteration_experiment import PruningIterationExperiment
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import PruningMethod
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder
from dmp.task.experiment.training_experiment import (
    training_experiment_keys,
    training_experiment_summarizer,
)
from dmp.task.experiment.training_experiment.experiment_record_settings import (
    RunSpecificConfig,
)
from dmp.task.experiment.training_experiment.model_saving_callback import (
    ModelSavingCallback,
)
from dmp.task.experiment.training_experiment.model_state_resume_config import ModelStateResumeConfig
from dmp.task.experiment.training_experiment.resume_config import ResumeConfig
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
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
from dmp.task.task import Task
from dmp.task.task_result import TaskResult

from dmp.worker import Worker


'''

TODO:
    + how to record seed-change epoch
    + writing child jobs into database
    + making sure model save & resume behavior is consistent
    + parent experiment history appending
    + model save custom paths?
    + test run
    + first batch

    [.8^(2) = .64 (36%), .8 (20%), .8^(1/2)~=.894 (10.6%), .8^(1/4) ~= .945 (5.4%)] pruning per IMP iteration 
        to target of <3.5% LeNet (16 iters), 3.5% ResNet (16 iters), 0.6% (24 iters) VGG:
    
    -> LTHChainExperiment: Make an ordinary training run, saving at critical save points.
        -> Save as if a normal training experiment?
            -> and mark run as a LTHChainExperiment
    -> For alternate seeds:
        -> LTHChainSeedChangeExperiment: finish training run with alternate seed
        -> for each pruning config, rewind point:
            -> PruningIterationExperiment: dispatch iterative pruning job sequence
                -> load trained network, mask
                -> prune trained network to get new mask
                -> load rewind point weights (but not mask)
                -> train & save
                -> dispatch next iteration job
    -> For main seed:
        -> for each pruning config, rewind point:
            -> PruningIterationExperiment: dispatch iterative pruning job sequence
                -> load trained network, mask
                -> prune trained network to get new mask
                -> load rewind point weights (but not mask)
                -> train & save
                -> dispatch next iteration job
    Other concerns:
        + tracking and labeling related runs:
            -> use parent_run id for overall run
            -> there's like:
                -> root_run: the root, originating training run 
                -> pruning_method: the pruning run sequence stemming from the root run
                -> iteration_number: this particular run/iteration of that sequence
            -> rewind_point: use rewind epoch data to identify rewind point epoch
            -> use seed / parent seed (?) to identify seed changes
                -> could also use a seed column?
        + aggregating as experiments?
            + parent run settings
            + rewind point settings
                + seed change
            + pruning settings
            + pruning iteration number
        + appending/combining history data
            + could load and append
                + load parent run record
                + append to run record
                + save as my run record
                * Convienient & fast for loading total history
                - Inefficient space usage
                - Handling branching records is awkward

'''


@dataclass
class LTHChainExperiment(DelegatingExperiment):
    baseline_experiment: TrainingExperiment
    pruning_configs: List[PruningConfig]
    pruning_seeds: List[int]
    rewind_epochs: List[int]

    @property
    def version(self) -> int:
        return super().version + 1

    def __call__(self, worker: Worker, job: Job,
                 *args,
                 **kwargs,
                 ) -> ExperimentResultRecord:

        result_record = self.baseline_experiment(worker, job, *args, **kwargs)
        # TODO: add this experiment's flags, etc to the result record

        from dmp.marshaling import marshal

        child_jobs = []
        for rewind_epoch in self.rewind_epochs:
            rewind_config = ModelStateResumeConfig(
                run_id=job.id,
                load_mask=False,
                load_optimizer=True,
                epoch=TrainingEpoch(
                    epoch=rewind_epoch,
                    model_number=0,
                    model_epoch=rewind_epoch,
                ),
            )

            for pruning_config in self.pruning_configs:
                for seed in self.pruning_seeds:
                    # TODO: enqueue prune and train experiments
                    if seed == self.baseline_experiment.seed:
                        # queue up a PruningIterationExperiment
                        child_experiment = PruningIterationExperiment(
                            **vars(self.baseline_experiment),
                            pruning=dataclass.replace(
                                pruning_config,
                                pruning_iteration=pruning_config.pruning_iteration+1,
                            ),
                            rewind=rewind_config,
                        )
                    else:
                        child_experiment = LTHSeedChangeExperiment(
                            **vars(self.baseline_experiment),
                            pruning=pruning_config,
                            rewind=rewind_config,
                        )

                    child_id = uuid.uuid4()
                    child_experiment.record = dataclass.replace(
                        child_experiment.record,
                        root_run=job.id,
                        parent_run=child_id,
                    )
                    child_job = Job(
                        id=child_id,
                        parent=job.id,
                        priority=job.priority -
                        1 if isinstance(job.priority, int) else job.priority,
                        command=marshal.marshal(child_experiment),
                    )
                    child_jobs.append(child_job)
                    

        return result_record
