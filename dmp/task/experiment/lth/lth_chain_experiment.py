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
from dmp.task.experiment.lth.lth_seed_change_experiment import LTHSeedChangeExperiment
from dmp.task.experiment.pruning_experiment.pruning_iteration_experiment import (
    PruningIterationExperiment,
)
from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.recorder.test_set_recorder import TestSetRecorder
from dmp.task.experiment.training_experiment import (
    training_experiment_keys,
    training_experiment_summarizer,
)
from dmp.task.experiment.run_spec import (
    RunSpec,
)
from dmp.task.experiment.training_experiment.model_saving_callback import (
    ModelSavingCallback,
)
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)
from dmp.task.experiment.training_experiment.epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
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
from dmp.task.experiment.experiment import ExperimentTask
from dmp.task.experiment.training_experiment.test_set_info import TestSetInfo
from dmp.model.model_info import ModelInfo

from dmp.dataset.dataset_spec import DatasetSpec
from dmp.model.model_spec import ModelSpec
from dmp.task.task import Task
from dmp.task.task_result import TaskResult

from dmp.context import Context


"""

DONE:
    + writing child jobs into database
    + parent experiment history appending

TODO:
    + refactor tasks in DB:
        + record.resume_from -> resume_from
    + saving model -- do we need model as param in restore_checkpoint?
    + making sure model save & resume behavior is consistent

    + move run data into run attrs
    + change attr/logging queries to save attrs to run table, ignore experiment
    + add task & query that assigns/creates experiment id's

    + checkpointing with resuming
    + standard "root" runs
    + pruning runs
        + including reps with different seed / data order
            + seed change reps need to complete first training run before pruning

    + how to record seed-change epoch -> add a column and make sure it is extended
    + model save custom paths?
    + test run
    + first batch

     + experiment / run refactor
        + all attrs go to run
            + kind is canonicalized too
            + or maybe simple jsonb indexing?
        + use attrs to select & aggregate experiment id's later

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

"""


@dataclass
class LTHChainExperiment(TrainingExperiment):
    pruning_configs: List[PruningConfig]
    pruning_seeds: List[int]
    rewind_epochs: List[int]

    @property
    def version(self) -> int:
        return super().version + 1

    # @property
    # def delegate(self) -> TrainingExperiment:
    #     return self.baseline_experiment

    def __call__(
        self,
        context: Context,
    ) -> ExperimentResultRecord:
        # NB: must have a compatible save mode
        self.record.save_model_epochs.append(self.rewind_epochs)  # type: ignore

        result_record = super()(context)
        # TODO: add this experiment's flags, etc to the result record

        from dmp.marshaling import marshal

        child_kwargs = vars(self)
        for key in ("pruning_configs", "pruning_seeds", "rewind_epochs"):
            del child_kwargs[key]

        child_tasks = []
        for rewind_epoch in self.rewind_epochs:
            rewind_config = TrainingExperimentCheckpoint(
                run_id=context.id,
                load_mask=False,
                load_optimizer=True,
                epoch=TrainingEpoch(
                    epoch=rewind_epoch,
                    model_number=0,
                    model_epoch=rewind_epoch,
                ),
            )

            # generate prune and train experiments
            for pruning_config in self.pruning_configs:
                for seed in self.pruning_seeds:
                    if seed == self.seed:
                        # no seed change: use a normal pruning iteration experiment
                        child_task = PruningIterationExperiment(
                            **child_kwargs,
                            prune=dataclass.replace(
                                pruning_config,
                                pruning_iteration=pruning_config.iteration + 1,
                            ),
                            rewind=rewind_config,
                        )
                    else:
                        child_task = LTHSeedChangeExperiment(
                            **child_kwargs,
                            pruning=pruning_config,
                            rewind=rewind_config,
                        )

                    child_id = uuid.uuid4()
                    child_task.seed = seed
                    child_task.record = dataclass.replace(
                        child_task.record,
                        root_run=context.id,
                        parent_run=context.id,
                        sequence_run=child_id,
                    )
                    child_tasks.append(child_task)

        # enqueue prune and train experiments
        context.push_tasks(child_tasks)

        return result_record
