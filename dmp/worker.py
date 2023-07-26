from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid
import tensorflow
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue
from dmp import common

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.postgres_interface.schema.postgres_schema import PostgresSchema


@dataclass
class Worker:
    _job_queue: JobQueue
    _schema: "PostgresSchema"
    # _result_logger: "ExperimentResultLogger"
    _strategy: tensorflow.distribute.Strategy
    _info: Dict[str, Any]
    _max_jobs: Optional[int] = None

    @property
    def strategy(self) -> tensorflow.distribute.Strategy:
        return self._strategy

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def schema(self) -> "PostgresSchema":
        return self._schema

    @property
    def job_queue(self) -> JobQueue:
        return self._job_queue

    def __call__(self):
        git_hash = common.get_git_hash()
        self._job_queue.work_loop(
            lambda worker_id, job: self._handler(worker_id, job, git_hash)
        )

    def _handler(
        self,
        worker_id: uuid.UUID,
        job: Job,
        git_hash: Optional[str],
    ) -> bool:
        from dmp.marshaling import marshal
        from dmp.task.task import Task

        # from dmp.task.experiment.experiment_result_record import ExperimentResultRecord
        from dmp.context import Context

        """
        + save resume task
        + save history
        + save model
        ---
        + load resume task
        + resume from saved history
        + resume from saved model
        """

        second_git_hash = common.get_git_hash()
        if second_git_hash is not git_hash and second_git_hash != git_hash:
            return False

        self._info["worker_id"] = worker_id

        # demarshal task from job.command
        task: Task = marshal.demarshal(job.command)  # type: ignore

        # run task
        with self.strategy.scope():
            result = task(Context(self, job, task))

        # log task run
        if isinstance(result, ExperimentResultRecord):
            self._result_logger.log(result)

        if self._max_jobs is not None:
            self._max_jobs -= 1

        second_git_hash = common.get_git_hash()
        return (self._max_jobs is None or self._max_jobs > 0) and (
            git_hash is second_git_hash or git_hash == second_git_hash
        )


# from dmp.logging.experiment_result_logger import ExperimentResultLogger
# from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
