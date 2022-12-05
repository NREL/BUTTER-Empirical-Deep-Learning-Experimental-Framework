from typing import Tuple, Type
from dmp.task.aspect_test.aspect_test_task import AspectTestTask
from dmp.task.training_experiment.training_experiment import TrainingExperiment
from dmp.task.growth_experiment.growth_experiment import GrowthExperiment

# register task types here
task_types: Tuple[Type, ...] = (
    AspectTestTask,
    TrainingExperiment,
    GrowthExperiment,
)
