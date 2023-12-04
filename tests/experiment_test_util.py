from typing import Optional
from uuid import UUID, uuid4
from dmp.context import Context
import dmp.jobqueue_interface.worker
import sys

from jobqueue.job import Job

from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.worker import Worker
from jobqueue.connect import load_credentials

sys.path.insert(0, "./")


def run_experiment(run, use_database: bool = False, id: Optional[UUID] = None):
    strategy = dmp.jobqueue_interface.worker.make_strategy(None, None, None)
    schema = None
    if id is None:
        id = uuid4()
    if use_database:
        credentials = load_credentials("dmp")
        schema = PostgresInterface(credentials)

    worker = Worker(
        None,  # type: ignore
        schema,  # type: ignore
        strategy,  # type: ignore
        {},  # type: ignore
    )  # type: ignore

    context = Context(worker, Job(id), run)
    run(context)
