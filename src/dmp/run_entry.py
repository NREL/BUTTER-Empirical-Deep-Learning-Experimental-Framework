import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from functools import total_ordering
import pandas
from dmp.task.run_status import RunStatus

from dmp.task.run_command import RunCommand


@total_ordering
@dataclass
class RunEntry:
    queue: int
    status: RunStatus
    priority: int
    id: uuid.UUID
    start_time: Optional[datetime]
    update_time: Optional[datetime]
    worker_id: Optional[uuid.UUID]
    parent_id: Optional[uuid.UUID]
    experiment_id: Optional[uuid.UUID]
    command: RunCommand
    history: Optional[pandas.DataFrame]
    extended_history: Optional[pandas.DataFrame]
    error_message: Optional[str]

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.id < other.id

    def __hash__(self):
        return self.id.__hash__()
