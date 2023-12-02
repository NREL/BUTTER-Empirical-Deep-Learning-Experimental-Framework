import os

from jobqueue.job import Job
from dmp.context import Context
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.postgres_interface.update_experiment_summary import UpdateExperimentSummary
from dmp.postgres_interface.update_experiment_summary_result import (
    UpdateExperimentSummaryResult,
)

from dmp.worker import Worker

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from jobqueue.connection_manager import ConnectionManager
import argparse
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas
import traceback
import json

from pprint import pprint
import uuid
from psycopg import sql

from jobqueue import load_credentials


import pathos.multiprocessing as multiprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_workers", type=int)
    parser.add_argument("block_size", type=int)
    parser.add_argument("lock_limit", type=int)
    args = parser.parse_args()

    num_workers = args.num_workers
    block_size = args.block_size
    lock_limit = args.lock_limit

    pool = multiprocessing.ProcessPool(num_workers)
    results = pool.uimap(
        do_work, ((i, block_size, lock_limit) for i in range(num_workers))
    )
    aggregate_result = UpdateExperimentSummaryResult(0, 0)
    for result in results:
        aggregate_result += result
    print(
        f"Done. Summarized {aggregate_result.num_experiments_updated} experiments, excepted {aggregate_result.num_experiments_excepted}."
    )
    pool.close()
    pool.join()
    print("Complete.")


def do_work(args) -> UpdateExperimentSummaryResult:
    worker_number, block_size, lock_limit = args
    # block_size = args[1]

    context = Context(
        Worker(
            None,  # type: ignore
            PostgresSchema(load_credentials("dmp")),
            None,  # type: ignore
            {},
            None,
        ),
        Job(uuid.uuid4()),
        UpdateExperimentSummary(
            block_size,
            lock_limit,
        ),
    )

    num_tries = 0
    worker_result = UpdateExperimentSummaryResult(0, 0)
    while num_tries < 32:
        this_result: UpdateExperimentSummaryResult = context.task(context)  # type: ignore
        worker_result = worker_result + this_result
        print(f"Updated {this_result}; lifetime total {worker_result}")
        if worker_result.num_experiments_updated == 0:
            num_tries += 1
        else:
            num_tries = 0

    return worker_result


if __name__ == "__main__":
    main()
