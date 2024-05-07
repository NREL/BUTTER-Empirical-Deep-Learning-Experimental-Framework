from dataclasses import dataclass

from dmp.task.run_result import RunResult


@dataclass
class UpdateExperimentSummaryResult(RunResult):
    num_experiments_updated: int
    num_experiments_excepted: int

    def __add__(
        self, other: "UpdateExperimentSummaryResult"
    ) -> "UpdateExperimentSummaryResult":
        return UpdateExperimentSummaryResult(
            self.num_experiments_updated + other.num_experiments_updated,
            self.num_experiments_excepted + other.num_experiments_excepted,
        )
