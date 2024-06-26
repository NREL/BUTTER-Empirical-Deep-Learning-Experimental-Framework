from copy import copy
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple


from dmp.task.experiment.lth.pruning_config import PruningConfig
from dmp.task.experiment.model_saving.model_saving_spec import ModelSavingSpec
from dmp.task.experiment.pruning_experiment.iterative_pruning_experiment import (
    IterativePruningExperiment,
)

from dmp.task.experiment.pruning_experiment.pruning_run_spec import (
    IterativePruningConfig,
)
from dmp.task.experiment.training_experiment.run_spec import RunConfig
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
        return 3

    def __call__(
        self,
        context: Context,
        config: RunConfig,
    ) -> None:
        # make sure rewind and resume points will be recorded
        if config.model_saving is None:
            config.model_saving = ModelSavingSpec(
                save_initial_model=True,
                save_trained_model=True,
                save_fit_epochs=[],
                save_epochs=[],
                fixed_interval=0,
                fixed_threshold=0,
                exponential_rate=0,
            )

        config.model_saving.save_trained_model = True
        save_epochs = set(config.model_saving.save_epochs)
        save_epochs.update(
            (config.rewind_epoch.epoch for config in self.pruning_configs)
        )
        config.model_saving.save_epochs = sorted(save_epochs)

        epoch_counter, experiment_history = super().__call__(context, config)

        base_experiment_config = copy(vars(self))
        del base_experiment_config["pruning_configs"]
        del base_experiment_config["num_additional_seeds_per_config"]
        # del base_experiment_config['data']

        base_run_config = copy(vars(config))
        # del base_run_config["seed"]
        del base_run_config["resume_checkpoint"]
        base_run_config["saved_models"] = []
        # del base_run_config['data']

        child_tasks = []

        for i in range(self.num_additional_seeds_per_config + 1):
            base_run_config["seed"] = config.seed + i
            for pruning_config in self.pruning_configs:
                child_pruning_config = copy(pruning_config)

                if i == 0:
                    child_pruning_config.new_seed = False
                    prune_first_iteration = True

                    resume_checkpoint = TrainingExperimentCheckpoint(
                        run_id=context.id,
                        load_mask=False,
                        load_optimizer=False,
                        epoch=epoch_counter.current_epoch,
                    )
                else:
                    child_pruning_config.new_seed = True
                    prune_first_iteration = False

                    resume_checkpoint = TrainingExperimentCheckpoint(
                        run_id=context.id,
                        load_mask=False,
                        load_optimizer=True,
                        epoch=pruning_config.rewind_epoch,
                    )

                child_tasks.append(
                    Run(
                        experiment=IterativePruningExperiment(
                            **base_experiment_config,
                            pruning=child_pruning_config,
                        ),
                        config=IterativePruningConfig(
                            **base_run_config,
                            resume_checkpoint=resume_checkpoint,
                            # seed=seed,
                            rewind_run_id=context.id,
                            prune_first_iteration=prune_first_iteration,
                        ),
                    )
                )

        context.push_runs(child_tasks)
