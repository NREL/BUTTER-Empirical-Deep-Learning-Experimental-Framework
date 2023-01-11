from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Sequence

from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.ml_task import MLTask
from dmp.task.training_experiment.training_experiment_keys import TrainingExperimentKeys


@dataclass
class Dataset():
    ml_task: MLTask
    train: Optional[DatasetGroup] = None
    test: Optional[DatasetGroup] = None
    validation: Optional[DatasetGroup] = None

    @property
    def splits(self) -> Sequence[Tuple[str, DatasetGroup]]:
        splits = []

        def try_add(key, value):
            if value is not None:
                splits.append((key, value))

        try_add(TrainingExperimentKeys.train, self.train)
        try_add(TrainingExperimentKeys.test, self.test)
        try_add(TrainingExperimentKeys.validation, self.validation)
        return splits

    @property
    def input_shape(self) -> List[int]:
        return [int(i) for i in self.train.inputs.shape[1:]]  # type: ignore

    @property
    def output_shape(self) -> List[int]:
        return [int(i) for i in self.train.outputs.shape[1:]]  # type: ignore
