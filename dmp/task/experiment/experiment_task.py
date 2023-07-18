from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from dmp.task.experiment.a_experiment_task import (
    AExperimentTask,
    ParameterDict,
    ParameterValue,
)
from dmp.task.task import Task
from dmp.common import keras_type_key, marshal_type_key


@dataclass
class ExperimentTask(AExperimentTask):
    batch: str  # the batch of experiments this one belongs to
    experiment_tags: Optional[
        Dict[str, ParameterValue]
    ]  # extra tags related to this experiment
    run_tags: Optional[
        Dict[str, ParameterValue]
    ]  # extra tags related to this run of this experiment
