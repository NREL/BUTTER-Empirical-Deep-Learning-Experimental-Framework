from typing import Iterable, List
from uuid import uuid4

from jobqueue.connect import load_credentials
from dmp.postgres_interface.schema.postgres_interface import PostgresInterface

from dmp.run_entry import RunEntry
from dmp.task.run import Run
from dmp.task.run_status import RunStatus


def enqueue_batch_of_runs(runs: List[Run], queue_id: int, base_priority: int):
    run_entries = [
        RunEntry(
            queue=queue_id,
            status=RunStatus.Queued,
            priority=base_priority + i,
            id=uuid4(),
            start_time=None,
            update_time=None,
            worker_id=None,
            parent_id=None,
            experiment_id=None,
            command=run,
            history=None,
            extended_history=None,
            error_message=None,
        )
        for i, run in enumerate(runs)
    ]

    print(f"Generated {len(run_entries)} jobs.")
    # pprint(jobs)
    credentials = load_credentials("dmp")
    database = PostgresInterface(credentials)
    database.push_runs(run_entries)
    print(f"Enqueued {len(run_entries)} jobs.")
