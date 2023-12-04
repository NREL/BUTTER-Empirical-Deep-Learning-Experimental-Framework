from enum import IntEnum
from jobqueue.job_status import JobStatus


class RunStatus(IntEnum):
    Failed = -1
    Queued = 0
    Claimed = 1
    Complete = 2
    Summarized = 3
