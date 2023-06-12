from dataclasses import dataclass, field
import dataclasses
import io
from pprint import pprint
from typing import List, Optional, Any, Dict, Tuple

from jobqueue.job import Job
import numpy
from dmp import parquet_util
from dmp.common import KerasConfig
import dmp.keras_interface.model_serialization as model_serialization
from dmp.layer.layer import Layer

from dmp.keras_interface.keras_utils import make_keras_instance, make_keras_kwcfg
from dmp.task.experiment.experiment_result_record import ExperimentResultRecord

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
import dmp.keras_interface.access_model_parameters as access_model_parameters

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.training_experiment.model_state_resume_config import ModelStateResumeConfig
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.task.task import Task
from dmp.worker import Worker


@dataclass
class PruneAndTrainExperiment(TrainingExperiment):
    """
    -> Make an ordinary training run, saving at critical save points.
    -> For each pruning configuration, at each rewind save point
        [.8^(2) = .64 (36%), .8 (20%), .8^(1/2)~=.894 (10.6%), .8^(1/4) ~= .945 (5.4%)] pruning per IMP iteration 
        to target of <3.5% LeNet (16 iters), 3.5% ResNet (16 iters), 0.6% (24 iters) VGG:

        -> Dispatch two IMP runs at specific save points as rewind points
            []
            -> One run uses same data order/seed
            -> The other run uses a different data order/seed
            -> Each IMP Run:
                -> optionally prune using pruning weights (different from rewind weights) 
                -> load rewind weights and optimizer
                -> train & save
                -> when completed, possibly dispatch a new pruning run 
                    -> use same rewind point, but new pruning weights
    """
    pruning_iteration: int
    pruning_max_iteration: int  # run-specific attribute?
    pruning_method: PruningMethod
    # TODO: what is run vs experiment attributes here?
    rewind_point: ModelStateResumeConfig

    # training_experiment: TrainingExperiment

    @property
    def version(self) -> int:
        return 1

    def __call__(
        self,
        worker: Worker,
        job: Job,
        *args,
        **kwargs,
    ) -> ExperimentResultRecord:
        with worker.strategy.scope():
            self._set_random_seeds()
            dataset, metrics = self._load_and_prepare_dataset()

            network = self._make_network(self.model)
            model = self._make_model_from_network(network, metrics)
            self._resume_model(model, self.record.resume_from)
            print(model.network.structure.summary())
            model.keras_model.summary()
            history = self._fit_model(
                worker,
                job,
                self.fit,
                dataset,
                model,
                [self._make_early_stopping_callback()],
                num_free_parameters=num_free_parameters,
                # new_model_number=True, # TODO: but not if this is the first iteration!
            )

            # TODO: if desired, enqueue next pruning/training iteration

            
            # # load prune source
            # # do pruning to get new mask
            # num_pruned = self.pruning_method.prune(
            #     model.network.structure, model.keras_network.layer_to_keras_map)
            # num_free_parameters = model.network.num_free_parameters - num_pruned

            # set weights from reset source
            # self._resume_model(model, self.rewind_point)

            return self._make_result_record(
                worker.worker_info,
                job.id,
                dataset,
                model.network,
                history,
            )

