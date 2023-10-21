from copy import copy
from dataclasses import dataclass, field
import io
from pprint import pprint
from typing import List, Optional, Any, Dict, Tuple

from jobqueue.job import Job
import numpy
from dmp import parquet_util
from dmp.common import KerasConfig
import dmp.keras_interface.model_serialization as model_serialization
from dmp.layer.layer import Layer

from dmp.keras_interface.keras_utils import make_keras_instance, keras_kwcfg
from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.pruning_experiment.iterative_pruning_experiment import (
    IterativePruningExperiment,
)

from dmp.task.experiment.pruning_experiment.pruning_method.pruning_method import (
    PruningMethod,
)
from dmp.task.experiment.pruning_experiment.pruning_run_spec import (
    IterativePruningRunSpec,
)
from dmp.task.experiment.training_experiment.run_spec import RunSpec
from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)
from dmp.context import Context
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)
from dmp.task.run import Run


@dataclass
class LTHExperiment(TrainingExperiment):
    pruning_configs: List[PruningConfig]
    num_additional_seeds_per_config: int

    @property
    def version(self) -> int:
        return 1

    def __call__(
        self,
        context: Context,
        run: RunSpec,
    ) -> None:
        # make sure rewind and resume points will be recorded
        if run.model_saving is None:
            run.model_saving = ModelSavingSpec(
                True,
                False,
                [],
                [],
                0,
                0,
                0,
            )

        run.model_saving.save_trained_model = False
        save_epochs = set(run.model_saving.save_epochs)
        save_epochs.update(
            (config.rewind_epoch.epoch for config in self.pruning_configs)
        )
        run.model_saving.save_epochs = sorted(save_epochs)

        epoch_counter, experiment_history = super().__call__(context, run)

        base_experiment_config = copy(vars(self))
        del base_experiment_config["pruning_configs"]
        del base_experiment_config["num_additional_seeds_per_config"]
        # del base_experiment_config['data']

        base_run_config = copy(vars(run))
        del base_run_config["seed"]
        del base_run_config["resume_checkpoint"]
        base_run_config["saved_models"] = []
        # del base_run_config['data']

        child_tasks = []

        for i in range(self.num_additional_seeds_per_config + 1):
            seed = run.seed + i
            for pruning_config in self.pruning_configs:
                child_pruning_config = copy(pruning_config)

                if i == 0:
                    child_pruning_config.new_seed = False
                    prune_first_iteration = True

                    resume_checkpoint = TrainingExperimentCheckpoint(
                        run_id=context.id,
                        load_mask=True,
                        load_optimizer=True,
                        epoch=epoch_counter.training_epoch,
                    )
                else:
                    child_pruning_config.new_seed = True
                    prune_first_iteration = False

                    resume_checkpoint = TrainingExperimentCheckpoint(
                        run_id=context.id,
                        load_mask=True,
                        load_optimizer=True,
                        epoch=pruning_config.rewind_epoch,
                    )

                child_tasks.append(
                    Run(
                        experiment=IterativePruningExperiment(
                            **base_experiment_config,
                            pruning=child_pruning_config,
                        ),
                        run=IterativePruningRunSpec(
                            **base_run_config,
                            resume_checkpoint=resume_checkpoint,
                            seed=seed,
                            rewind_run_id=context.id,
                            prune_first_iteration=prune_first_iteration,
                        ),
                    )
                )

        context.push_tasks(child_tasks)
