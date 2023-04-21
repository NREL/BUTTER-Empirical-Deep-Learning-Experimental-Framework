from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Sequence

from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.ml_task import MLTask
import dmp.task.experiment.training_experiment.training_experiment_keys as training_experiment_keys


@dataclass
class Dataset():
    '''
    Represents a ML dataset.
    '''

    ml_task: MLTask # task that this dataset is intended for
    train: Optional[DatasetGroup] = None
    test: Optional[DatasetGroup] = None
    validation: Optional[DatasetGroup] = None

    @property
    def splits(self) -> Sequence[Tuple[str, DatasetGroup]]:
        splits = []

        def try_add(key, value):
            if value is not None:
                splits.append((key, value))

        try_add(training_experiment_keys.keys.train, self.train)
        try_add(training_experiment_keys.keys.test, self.test)
        try_add(training_experiment_keys.keys.validation, self.validation)
        return splits

    @property
    def full_splits(self)->Sequence[Tuple[str, Optional[DatasetGroup]]]:
        return (
                (training_experiment_keys.keys.train, self.train),
                (training_experiment_keys.keys.test, self.test),
                (training_experiment_keys.keys.validation, self.validation),
            )

    @property
    def input_shape(self) -> List[int]:
        return [int(i) for i in self.train.inputs.shape[1:]]  # type: ignore

    @property
    def output_shape(self) -> List[int]:
        return [int(i) for i in self.train.outputs.shape[1:]]  # type: ignore
