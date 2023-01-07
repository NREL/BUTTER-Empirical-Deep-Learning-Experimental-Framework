from dataclasses import dataclass
from typing import Dict, List, Optional
import uuid
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue
from dmp.logging.result_logger import ResultLogger
from dmp.task.task import Task
from dmp.task.task_result_record import TaskResultRecord


from lmarshal.src.marshal import Marshal
import tensorflow


@dataclass
class Worker:
    _job_queue: JobQueue
    _result_logger: ResultLogger
    _strategy: tensorflow.distribute.Strategy
    _worker_info: Dict
    _max_jobs: Optional[int] = None

    @property
    def strategy(self) -> tensorflow.distribute.Strategy:
        return self._strategy

    @property
    def worker_info(self) -> Dict:
        return self._worker_info

    def __call__(self):
        self._job_queue.work_loop(
            lambda worker_id, job: self._handler(worker_id, job))

    def _handler(self, worker_id: uuid.UUID, job: Job) -> bool:
        from dmp.jobqueue_interface import jobqueue_marshal
        
        # demarshal task from job.command
        task: Task = jobqueue_marshal.demarshal(job.command)

        # run task
        result : TaskResultRecord = task(self)

        # log task run
        self._result_logger.log(
            result,
            job.id,
            worker_id,
        )

        if self._max_jobs is None:
            return True

        self._max_jobs -= 1
        return self._max_jobs > 0
