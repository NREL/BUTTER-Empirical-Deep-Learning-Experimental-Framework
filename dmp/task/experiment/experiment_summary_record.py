from dataclasses import dataclass
import uuid


@dataclass
class ExperimentSummaryRecord():
    experiment_uid: uuid.UUID
    core_data: bytes
    extended_data: bytes
