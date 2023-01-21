from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid
import tensorflow
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue

from dmp.task.experiment.experiment_result_record import ExperimentResultRecord


@dataclass
class Worker:
    _job_queue: JobQueue
    _result_logger: 'ExperimentResultLogger'
    _strategy: tensorflow.distribute.Strategy
    _worker_info: Dict[str, Any]
    _max_jobs: Optional[int] = None

    @property
    def strategy(self) -> tensorflow.distribute.Strategy:
        return self._strategy

    @property
    def worker_info(self) -> Dict[str, Any]:
        return self._worker_info

    def __call__(self):
        self._job_queue.work_loop(
            lambda worker_id, job: self._handler(worker_id, job))

    def _handler(self, worker_id: uuid.UUID, job: Job) -> bool:
        from dmp.marshaling import marshal
        from dmp.task.task import Task

        self._worker_info['worker_id'] = worker_id

        # demarshal task from job.command
        task: Task = marshal.demarshal(job.command) # type: ignore

        # run task
        result = task(self, job)

        # log task run
        if isinstance(result, ExperimentResultRecord):
            self._result_logger.log(result)

        if self._max_jobs is None:
            return True

        self._max_jobs -= 1
        return self._max_jobs > 0


from dmp.logging.experiment_result_logger import ExperimentResultLogger