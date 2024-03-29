from dataclasses import dataclass
from typing import Optional
import pandas


@dataclass
class ExperimentSummaryRecord:
    num_runs: int
    by_epoch: Optional[pandas.DataFrame]
    by_loss: Optional[pandas.DataFrame]
