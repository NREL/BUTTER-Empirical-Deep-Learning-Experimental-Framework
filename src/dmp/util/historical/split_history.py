from itertools import chain
import os
import jobqueue

from psycopg import ClientCursor
import pyarrow
from dmp.common import flatten
from dmp.postgres_interface.element.column_group import ColumnGroup

from dmp.postgres_interface.schema.postgres_interface import PostgresInterface
from dmp.task.experiment.training_experiment import training_experiment_keys
from dmp.task.experiment.training_experiment.training_experiment_keys import (
    TrainingExperimentKeys,
)

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
from psycopg.sql import SQL, Literal, Identifier
from dmp.postgres_interface.postgres_interface_common import sql_comma, sql_placeholder

from jobqueue.cursor_manager import CursorManager
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask
from dmp.layer.dense import Dense
from dmp.postgres_interface.postgres_compressed_result_logger import (
    PostgresCompressedResultLogger,
)

from dmp.logging.postgres_parameter_map_v1 import PostgresParameterMapV1
from dmp.model.dense_by_size import DenseBySize

from dmp.task.experiment.training_experiment.training_experiment import (
    TrainingExperiment,
)

from dmp.marshaling import marshal

import pathos.multiprocessing as multiprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_workers", type=int)
    parser.add_argument("block_size", type=int)
    args = parser.parse_args()

    num_workers = int(args.num_workers)
    block_size = int(args.block_size)

    pool = multiprocessing.ProcessPool(num_workers)
    results = pool.uimap(do_work, ((i, block_size) for i in range(num_workers)))
    total_num_converted = sum(results)
    print(f"Done. Converted {total_num_converted} runs.")
    pool.close()
    pool.join()
    print("Complete.")


def do_work(args):
    worker_number, block_size = args

    credentials = jobqueue.load_credentials("dmp")
    schema = PostgresInterface(credentials)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0
    print(f"Worker {worker_number} : {worker_id} started...")

    while True:  #  binary=True, scrollable=True
        num_converted = 0
        num_excepted = 0

        run = schema.run

        # run_id_column = run['values'].column('run_id')
        # history_column = run['history']
        # extended_history_column = run['extended_history']
        select_columns = ColumnGroup(
            run.run_id,
            run.run_history,
        )

        update_columns = ColumnGroup(select_columns, run.extended_history)

        get_and_lock_query = SQL(
            """
SELECT
    {columns}
FROM
    {run}
WHERE
    {extended_history} IS NULL
FOR UPDATE
SKIP LOCKED
LIMIT {block_size}
;"""
        ).format(
            columns=select_columns.columns_sql,
            run=run.identifier,
            extended_history=run.extended_history.identifier,
            block_size=Literal(block_size),
        )

        keys = training_experiment_keys.keys
        extended_cols = keys.extended_history_columns

        with ConnectionManager(schema.credentials) as connection:
            with connection.transaction():
                run_updates = []
                with connection.cursor(binary=True) as cursor:
                    cursor.execute(get_and_lock_query, binary=True)
                    rows = list(cursor.fetchall())
                    for row in rows:

                        def value(col):
                            return row[select_columns[col]]

                        run_id = value(run.run_id)
                        run_history: pandas.DataFrame = (
                            schema.convert_bytes_to_dataframe(value(run.run_history))
                        )  # type: ignore
                        extended_history = run_history.copy()

                        for k in list(run_history.columns):
                            if k in extended_cols:
                                run_history.drop(k, axis=1, inplace=True)
                            else:
                                extended_history.drop(k, axis=1, inplace=True)

                        extended_history[keys.epoch] = run_history[keys.epoch]

                        run_updates.append(
                            (
                                run_id,
                                schema.convert_dataframe_to_bytes(run_history),
                                schema.convert_dataframe_to_bytes(extended_history),
                            )
                        )
                        # raise Exception('asdf')
                        # except Exception as e:
                        #     num_excepted += 1
                        #     print(f'failed on Exception: {e}', flush=True)
                        #     traceback.print_exc()
                        #     # errors[experiment_id] = e

                    if len(run_updates) > 0:
                        cursor.execute(
                            SQL(
                                """
UPDATE {run} SET
    {history} = {values}.{history},
    {extended_history} = {values}.{extended_history}
FROM
    (VALUES {placeholders}) {values} ({update_columns})
WHERE
    {run}.{run_id} = {values}.{run_id}
                        ;"""
                            ).format(
                                run=run.identifier,
                                history=run.run_history.identifier,
                                extended_history=run.extended_history.identifier,
                                values=Identifier("_values"),
                                placeholders=sql_comma.join(
                                    [SQL("({})").format(update_columns.placeholders)]
                                    * len(run_updates)
                                ),
                                update_columns=update_columns.columns_sql,
                                run_id=run.run_id.identifier,
                            ),
                            list(chain(*run_updates)),
                            binary=True,
                        )
                    num_converted = len(run_updates)
        total_num_converted += num_converted
        total_num_excepted += num_excepted
        print(
            f"Worker {worker_number} : {worker_id} committed {num_converted}, excepted {num_excepted} runs. Lifetime total: {total_num_converted} / {total_num_excepted}."
        )

        if num_converted <= 0 and num_excepted <= 0:
            break

    return total_num_converted


if __name__ == "__main__":
    main()
