from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import UUID


from dmp.task.experiment.model_saving.model_saving_spec import (
    ModelSavingSpec,
)
from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch
from dmp.task.experiment.training_experiment.training_experiment_checkpoint import (
    TrainingExperimentCheckpoint,
)


@dataclass
class RunSpec:
    seed: int  # random seed to use

    data: dict  # run-specific information, e.g. runtime environment

    record_post_training_metrics: bool
    record_times: bool

    model_saving: Optional[ModelSavingSpec]
    saved_models: List[TrainingEpoch]

    # point to resume from (None means to start fresh)
    resume_checkpoint: Optional[TrainingExperimentCheckpoint]

    # checkpoint_interval: float = 30 * 60.0 # how often to checkpoint in seconds (if implemented)
