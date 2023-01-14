import os
from jobqueue.connection_manager import ConnectionManager
import psycopg

from tensorflow.python import traceback

from dmp.keras_interface.keras_utils import make_keras_config

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas
import io
from itertools import product
import json
import pathos.multiprocessing as multiprocessing

from pprint import pprint
import uuid
from psycopg import sql

import pyarrow
import pyarrow.parquet as parquet
from jobqueue import load_credentials
from jobqueue.cursor_manager import CursorManager
import numpy
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset
from dmp.layer.dense import Dense
from dmp.logging.postgres_compressed_result_logger import PostgresCompressedResultLogger

from dmp.logging.postgres_attribute_map import PostgresAttributeMap
import sys
from dmp.logging.postgres_parameter_map_v1 import PostgresParameterMapV1
from dmp.model.dense_by_size import DenseBySize

from dmp.parquet_util import make_pyarrow_schema
from dmp.task.training_experiment.training_experiment import TrainingExperiment

from dmp.marshaling import marshal

pmlb_index_path = os.path.join(
    os.path.realpath(os.path.join(
        os.getcwd(),
        os.path.dirname(__file__),
    )),
    'pmlb.csv',
)
dataset_index = pandas.read_csv(pmlb_index_path)
dataset_index.set_index('Dataset', inplace=True, drop=False)


@dataclass
class PsuedoPreparedDataset():
    ml_task: MLTask
    input_shape: List[int]
    output_shape: List[int]
    train_size: int
    test_size: int
    validation_size: int
    train: Any = None
    test: Any = None
    validation: Any = None


status_columns = [
    'id',
    'queue',
    'status',
    'priority',
    'start_time',
    'update_time',
    'worker',
    'error_count',
    'error',
]

data_columns = [
    'command',
    'parent',
]

columns = status_columns + data_columns

column_index_map = {name: i for i, name in enumerate(columns)}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('num_workers', type=int)
    parser.add_argument('block_size', type=int)
    args = parser.parse_args()

    num_workers = args.num_workers
    block_size = args.block_size

    pool = multiprocessing.ProcessPool(num_workers)
    results = pool.uimap(do_work,
                         ((i, block_size) for i in range(num_workers)))
    total_num_converted = sum(results)
    print(f'Done. Converted {total_num_converted} runs.')
    pool.close()
    pool.join()
    print('Complete.')


def do_work(args):
    worker_number, block_size = args

    credentials = load_credentials('dmp')

    result_logger = PostgresCompressedResultLogger(credentials)

    worker_id = str(worker_number) + str(uuid.uuid4())
    total_num_converted = 0
    total_num_excepted = 0
    print(f'Worker {worker_number} : {worker_id} started...')

    while True:  #  binary=True, scrollable=True
        num_converted = 0
        num_excepted = 0

        with ConnectionManager(credentials) as connection:
            with connection.transaction():
                # cursor.itersize = 8

                column_selection = sql.SQL(', ').join([
                    sql.SQL('s.{col} {col}').format(col=sql.Identifier(c))
                    for c in status_columns
                ] + [
                    sql.SQL('d.{col} {col}').format(col=sql.Identifier(c))
                    for c in data_columns
                ])

                q = sql.SQL("""
SELECT {column_selection}
FROM 
    (   SELECT *
        FROM job_status s
        WHERE 
            status = -10
        ORDER BY priority ASC
        FOR UPDATE
        SKIP LOCKED
        LIMIT {block_size}
    ) s
    INNER JOIN job_data d USING (id)
;""").format(
                    column_selection=column_selection,
                    block_size=sql.Literal(block_size),
                )
                with connection.cursor(binary=True) as cursor:
                    cursor.execute(q)

                    eids = set()
                    for row in cursor:
                        eids.add(row[column_index_map['id']])
                        try:
                            if convert_task(row, connection):
                                num_converted += 1
                            else:
                                num_excepted += 1
                        except Exception as e:
                            num_excepted += 1
                            print(f'failed on Exception: {e}')
                            traceback.print_exc()

                if len(eids) > 0:
                    eid_values = sql.SQL(',').join(
                        (sql.Literal(v) for v in sorted(eids)))
                    q = sql.SQL("""
UPDATE job_status
    SET status = 0
WHERE
    id IN ({eid_values})
                    ;""").format(eid_values=eid_values)
                    connection.execute(q)
        total_num_converted += num_converted
        total_num_excepted += num_excepted
        print(
            f'Worker {worker_number} : {worker_id} comitted {num_converted}, excepted {num_excepted} runs. Lifetime total: {total_num_converted} / {total_num_excepted}.'
        )

        if num_converted <= 0 and num_excepted <= 0:
            break

    return total_num_converted


def convert_task(row, connection) -> bool:

    def get_cell(column: str):
        return row[column_index_map[column]]

    def get_default(column: str, default):
        v = get_cell(column)
        if v is None:
            v = default
        return v

    src = get_cell('command')

    run_config = src.get('run_config')
    if src.get('', None) != 'AspectTestTask':
        return False

    def migrate_keras_config(target):
        if target is None:
            return None
        config = target.copy()
        keras_type = config.pop('type')
        return make_keras_config(
            keras_type,
            config,
        )  # type: ignore

    if src.get('early_stopping', None) is not None:
        return False

    experiment = TrainingExperiment(
        seed=src['seed'],
        batch=src['batch'],
        precision='float32',
        dataset=DatasetSpec(
            name=src['dataset'],
            source='pmlb',
            method=src.get('test_split_method', 'shuffled_train_test_split'),
            test_split=float(src.get('test_split', 0.2)),
            validation_split=0.0,
            label_noise=float(src.get('label_noise', 0.0)),
        ),
        model=DenseBySize(
            input=None,
            output=None,
            shape=src.get('shape'),
            size=src.get('size'),
            depth=src.get('depth'),
            search_method='integer',
            inner=Dense.make(
                -1, {
                    'activation':
                    src.get('activation', 'relu'),
                    'kernel_initializer':
                    'GlorotUniform',
                    'kernel_regularizer':
                    migrate_keras_config(src.get('kernel_regularizer', None)),
                    'bias_regularizer':
                    migrate_keras_config(src.get('bias_regularizer', None)),
                    'activity_regularizer':
                    migrate_keras_config(src.get('activity_regularizer',
                                                 None)),
                }),
        ),
        fit={
            'batch_size': run_config.get('batch_size'),
            'epochs': run_config.get('epochs'),
        },
        optimizer={
            'class': 'Adam',
            'learning_rate': 0.0001
        },
        loss=None,
        early_stopping=None,
        record_post_training_metrics=False,
        record_times=False,
        record_model=None,
        record_metrics=None,
    )

    new_command = psycopg.types.json.Jsonb(marshal.marshal(experiment))

    connection.execute(sql.SQL("""
UPDATE job_data d
    SET command = %b
WHERE "id" = %b
    """), (new_command, get_cell('id')),
                       binary=True)

    return True


if __name__ == "__main__":
    main()
