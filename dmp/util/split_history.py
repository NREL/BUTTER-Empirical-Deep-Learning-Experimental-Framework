from itertools import chain
import os

from psycopg import ClientCursor
import pyarrow
from dmp.common import flatten
from dmp.postgres_interface.column_group import ColumnGroup

from dmp.postgres_interface.postgres_schema import PostgresSchema
from dmp.task.experiment.training_experiment import training_experiment_keys
from dmp.task.experiment.training_experiment.training_experiment_keys import TrainingExperimentKeys

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

from jobqueue import load_credentials
from jobqueue.cursor_manager import CursorManager
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask
from dmp.layer.dense import Dense
from dmp.postgres_interface.postgres_compressed_result_logger import PostgresCompressedResultLogger

from dmp.logging.postgres_parameter_map_v1 import PostgresParameterMapV1
from dmp.model.dense_by_size import DenseBySize

from dmp.task.experiment.training_experiment.training_experiment import TrainingExperiment

from dmp.marshaling import marshal

import pathos.multiprocessing as multiprocessing


def main():
    # global schema, credentials

    parser = argparse.ArgumentParser()
    parser.add_argument('num_workers', type=int)
    parser.add_argument('block_size', type=int)
    args = parser.parse_args()

    num_workers = int(args.num_workers)
    block_size = int(args.block_size)

    pool = multiprocessing.ProcessPool(num_workers)
    results = pool.uimap(do_work,
                         ((i, block_size) for i in range(num_workers)))
    total_num_converted = sum(results)
    print(f'Done. Converted {total_num_converted} runs.')
    pool.close()
    pool.join()
    print('Complete.')


def do_work(args):
    global schema, credentials

    worker_number, block_size = args

    credentials = load_credentials('dmp')
    schema = PostgresSchema(credentials)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0
    print(f'Worker {worker_number} : {worker_id} started...')

    while True:  #  binary=True, scrollable=True
        num_converted = 0
        num_excepted = 0

        run = schema.run

        run_id_column = run['values'].column_as_group('run_id')
        history_column = run['history']
        extended_history_column = run['extended_history']
        columns = ColumnGroup.concatenate((
            run_id_column,
            history_column,
            extended_history_column,
        ))

        get_and_lock_query = SQL("""
SELECT 
    {columns}
FROM 
    {run}
WHERE 
    {extended_history} IS NULL
ORDER BY {run_id}
FOR UPDATE
SKIP LOCKED
LIMIT {block_size}
;""").format(
            columns=columns.columns_sql,
            run=run.identifier,
            extended_history=extended_history_column.identifier,
            run_id=run_id_column.identifier,
            block_size=Literal(block_size),
        )

        prefixes = training_experiment_keys.keys.data_set_prefixes

        extended_columns = list(chain(*[[
                p + c for c in training_experiment_keys.keys.extended_history_columns]
                              for p in prefixes]))
        with ConnectionManager(credentials) as connection:
            with connection.transaction():
                run_data = []
                with connection.cursor(binary=True) as cursor:

                    cursor.execute(get_and_lock_query, binary=True)
                    rows = list(cursor.fetchall())
                    for row in rows:
                        try:
                            run_id = row[columns['run_id']]
                            try:
                                source_history = schema.load_history_from_bytes(
                                    row[columns['run_history']])
                            except Exception e:


                            extended_history = None

                            ehc = {
                                k
                                for k in extended_columns
                                if k in source_history.columns
                            }

                            # print(f'ehc {ehc}')

                            extended_history = source_history[list(ehc)]
                            history = source_history[[
                                k for k in source_history.columns
                                if k not in ehc
                            ]]

                            # print(f'run {run_id}')
                            # print(history)
                            # print(extended_history)

                            run_data.append((
                                run_id,
                                schema.make_history_bytes(history),
                                schema.make_history_bytes(extended_history),
                            ))
                        except Exception as e:
                            num_excepted += 1
                            print(f'failed on Exception: {e}', flush=True)
                            traceback.print_exc()
                            # errors[experiment_id] = e

                    if len(run_data) > 0:
                        cursor.execute(
                            SQL("""
UPDATE {run} SET
    {history} = {values}.{history},
    {extended_history} = {values}.{extended_history}
FROM
    (VALUES {placeholders}) {values}({columns})
WHERE
    {run}.{run_id} = {values}.{run_id}
                        ;""").format(
                                run=run.identifier,
                                history=history_column.identifier,
                                extended_history=extended_history_column.
                                identifier,
                                values=Identifier('_values'),
                                placeholders=sql_comma.join(
                                    [SQL('({})').format(columns.placeholders)
                                     ] * len(run_data)),
                                columns=columns.columns_sql,
                                run_id=run_id_column.identifier,
                            ),
                            list(flatten(
                                run_data,
                                levels=1,
                            )),
                            binary=True,
                        )
                    num_converted = len(run_data)
        total_num_converted += num_converted
        total_num_excepted += num_excepted
        print(
            f'Worker {worker_number} : {worker_id} committed {num_converted}, excepted {num_excepted} runs. Lifetime total: {total_num_converted} / {total_num_excepted}.'
        )

        if num_converted <= 0 and num_excepted <= 0:
            break

    return total_num_converted


if __name__ == "__main__":
    main()
