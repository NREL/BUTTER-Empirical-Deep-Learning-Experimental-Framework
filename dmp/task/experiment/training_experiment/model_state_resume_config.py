from dataclasses import dataclass
from typing import Tuple
from uuid import UUID
from dmp.model.model_info import ModelInfo

from dmp.task.experiment.training_experiment.training_epoch import TrainingEpoch


@dataclass
class ModelStateResumeConfig:
    run_id: UUID
    load_mask: bool
    load_optimizer: bool
    epoch: TrainingEpoch

    def resume(
        self,
        model: ModelInfo,
    ) -> None:
        import dmp.keras_interface.model_serialization as model_serialization

        model_serialization.load_model_from_file(
            self.run_id,
            self.epoch.model_number,
            self.epoch.model_epoch,
            model,
            load_mask=self.load_mask,
            load_optimizer=self.load_optimizer,
        )

    def get_epoch(self) -> TrainingEpoch:
        return self.epoch
