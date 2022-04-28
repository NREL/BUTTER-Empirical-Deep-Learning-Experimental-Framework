from calendar import c
from sqlalchemy.dialects.postgresql import insert
from psycopg2.extensions import register_adapter, AsIs
import math
from pathos.multiprocessing import Pool
from dmp.logging.postgres_parameter_map import PostgresParameterMap
import jobqueue
from jobqueue.cursor_manager import CursorManager
import sqlalchemy
import json
import os
import time
import pandas as pd
import sys
from psycopg2 import sql
import numpy
import h5py

sys.path.append("../../")


credentials = jobqueue.connect.load_credentials('dmp')


def load_parameter_map():
    with CursorManager(credentials) as cursor:
        return PostgresParameterMap(cursor)


parameter_map = load_parameter_map()


def func():
    # num_threads = 64
    # num_threads = 1
    block_size = 10

    low_experiment_id = 0
    high_experiment_id = 10

    column_list = [
        'experiment_id',
        'update_timestamp',
        'experiment_parameters',
        'num_runs',
        'num',
        'val_loss_num_finite',
        'val_loss_avg',
        'val_loss_stddev',
        'val_loss_min',
        'val_loss_max',
        'val_loss_percentile',
        'loss_num_finite',
        'loss_avg',
        'loss_stddev',
        'loss_min',
        'loss_max',
        'loss_percentile',
        'val_accuracy_avg',
        'val_accuracy_stddev',
        'accuracy_avg',
        'accuracy_stddev',
        'val_mean_squared_error_avg',
        'val_mean_squared_error_stddev',
        'mean_squared_error_avg',
        'mean_squared_error_stddev',
        'val_kullback_leibler_divergence_avg',
        'val_kullback_leibler_divergence_stddev',
        'kullback_leibler_divergence_avg',
        'kullback_leibler_divergence_stddev',
        'num_free_parameters',
        'network_structure',
        'widths',
        'size',
        'relative_size_error',
    ]

    columns_sql = sql.SQL(',').join(map(sql.Identifier, column_list))

    column_map = {c: i for c, i in enumerate(column_list)}

    def fetch_block():
        with CursorManager(credentials) as cursor:
            cursor.itersize = block_size
            cursor.execute(sql.SQL("""
    SELECT
        {columns_sql}
    FROM
        experiment_summary_
    WHERE
        experiment_id >= %(low_experiment_id)s AND
        experiment_id < %(high_experiment_id)s
    ;""").format(
                columns_sql = columns_sql,
            ),
            dict(
                low_experiment_id=low_experiment_id,
                high_experiment_id=high_experiment_id,
            ))
        
            while True:
                row = cursor.fetchone()

                print(row)
                break
        return None

    fetch_block()
    # print(f'Initializing pool...')
    # start_time = time.perf_counter()
    # for i in range(0, num_readers):
    #     read_chunk(i)
    # with Pool(num_readers) as p:
    #     p.map(read_chunk, range(0, num_readers))
    # data_load = Parallel(n_jobs=num_readers, batch_size=1, backend='multiprocessing')(
    #     delayed(read_chunk(i)) for i in range(num_readers))

    # delta_t = time.perf_counter() - start_time
    # print(
    #     f'Processed {count} entries in {delta_t}s at a rate of {count / delta_t} entries / second.')


if __name__ == "__main__":
    func()
