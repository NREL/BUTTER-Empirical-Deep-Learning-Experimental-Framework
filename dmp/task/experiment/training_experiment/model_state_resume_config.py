from dataclasses import dataclass
from typing import Tuple
from uuid import UUID
from dmp.model.model_info import ModelInfo

from dmp.task.experiment.training_experiment.resume_config import ResumeConfig


@dataclass
class ModelStateResumeConfig(ResumeConfig):
    run_id: UUID
    epoch: int
    model_number: int
    model_epoch: int
    load_mask: bool
    load_optimizer: bool

    def resume(
        self,
        model: ModelInfo,
    ) -> None:
        import dmp.keras_interface.model_serialization as model_serialization

        model_serialization.load_model_from_file(
            self.run_id,
            self.model_number,
            self.model_epoch,
            model,
            load_mask=self.load_mask,
            load_optimizer=self.load_optimizer,
        )

    def get_epoch(self) -> Tuple[int, int, int]:
        return (
            self.epoch,
            self.model_number,
            self.model_epoch,
        )
