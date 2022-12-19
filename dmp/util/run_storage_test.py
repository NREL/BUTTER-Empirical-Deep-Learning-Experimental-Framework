import io
from itertools import product
import json
from multiprocessing import Pool
import os
from pprint import pprint
import uuid
from psycopg2 import sql

import pyarrow
import pyarrow.parquet as parquet
from jobqueue import *
from jobqueue.cursor_manager import CursorManager
import numpy
from sqlalchemy import column
from dmp.dataset.dataset_util import load_dataset

from dmp.logging.postgres_parameter_map import PostgresParameterMap
import sys

from dmp.parquet_util import make_pyarrow_schema


def main():

    credentials = load_credentials('dmp')
    parameter_map = None
    with CursorManager(credentials) as cursor:
        parameter_map = PostgresParameterMap(cursor)

    # columns
    # schema = pyarrow.schema(columns)

    # # use_dictionary=parameter_column_names,
    # # use_byte_stream_split=data_column_names,
    # use_byte_stream_split = [
    #     c.name for c in columns
    #     if c.type in [
    #         pyarrow.float32(), pyarrow.list_(pyarrow.float32()),
    #         pyarrow.float64(), pyarrow.list_(pyarrow.float64()),
    #     ]
    # ]

    # use_dictionary = [c.name for c in parameter_columns]

    info_columns = [
        'experiment_id',
        'run_id',
        'run_parameters',
    ]

    history_columns = [
        'test_loss',
        'train_loss',
        'test_accuracy',
        'train_accuracy',
        'test_loss',
        'train_loss',
        'test_accuracy',
        'train_accuracy',
        'test_mean_squared_error',
        'train_mean_squared_error',
        'test_mean_absolute_error',
        'train_mean_absolute_error',
        'test_root_mean_squared_error',
        'train_root_mean_squared_error',
        'test_mean_squared_logarithmic_error',
        'train_mean_squared_logarithmic_error',
        'test_hinge',
        'train_hinge',
        'test_squared_hinge',
        'train_squared_hinge',
        'test_cosine_similarity',
        'train_cosine_similarity',
        'test_kullback_leibler_divergence',
        'train_kullback_leibler_divergence',
    ]

    columns = info_columns + history_columns
    col_map = {name: i for i, name in enumerate(columns)}

    with CursorManager(credentials, name=str(uuid.uuid1()),
                       autocommit=False) as cursor:
        cursor.itersize = 8

        q = sql.SQL('SELECT ')
        q += sql.SQL(', ').join([
            sql.SQL('{} {}').format(sql.Identifier(c), sql.Identifier(c))
            for c in columns
        ])

        q += sql.SQL(' FROM run_ r ')
        q += sql.SQL(' ORDER BY record_timestamp DESC LIMIT 10')
        q += sql.SQL(' ;')

        x = cursor.mogrify(q)
        print(x)
        cursor.execute(q)
        shrinkage = []
        for row in cursor:

            src_parameters = {
                kind: value
                for kind, value in parameter_map.parameter_from_id(row[
                    col_map['run_parameters']])
            }

            pprint(src_parameters)

            dataset_src = 'pmlb'
            dataset :Dataset = load_dataset(dataset_src, src_parameters['dataset']) # type: ignore
            print(f'ml_task {dataset.ml_task}')
            
            result_block = {}
            for name in history_columns:
                v = row[col_map[name]]
                if v is not None and len(v) > 0:
                    result_block[name] = v

            if len(result_block) > 0:
                num_epochs = max(
                    (len(h) for h in result_block.values() if h is not None))
                for h in result_block.values():
                    h.extend([float('NaN')] * (num_epochs - len(h)))

                result_block['epoch'] = list(range(1, 1 + num_epochs))

                schema, use_byte_stream_split = make_pyarrow_schema(result_block.items())
                
                table = pyarrow.Table.from_pydict(result_block, schema=schema)

                buffer = io.BytesIO()
                pyarrow_file = pyarrow.PythonFile(buffer)

                parquet.write_table(
                    table,
                    pyarrow_file,
                    # root_path=dataset_path,
                    # schema=schema,
                    # partition_cols=partition_cols,
                    data_page_size=8 * 1024,
                    # compression='BROTLI',
                    # compression_level=8,
                    compression='ZSTD',
                    compression_level=12,
                    use_dictionary=False,
                    use_byte_stream_split=use_byte_stream_split,
                    version='2.6',
                    data_page_version='2.0',
                    # existing_data_behavior='overwrite_or_ignore',
                    # use_legacy_dataset=False,
                    write_statistics=False,
                    # write_batch_size=64,
                    # dictionary_pagesize_limit=64*1024,
                )
                bytes_written = buffer.getbuffer().nbytes
                # print(
                #     f'Written {bytes_written} bytes, {uncompressed} uncompressed, ratio {uncompressed/bytes_written}, shrinkage {bytes_written/uncompressed}, max relative error {round(numpy.max(max_relative_error)*100, 6)}%.'
                # )
                print(
                    f'Written {bytes_written} bytes.'
                )
                # shrinkage.append(bytes_written / uncompressed)

        shrinkage = numpy.array(shrinkage)
        print(
            f'average shrinkage: {numpy.average(shrinkage)} average ratio: {numpy.average(1.0/shrinkage)}'
        )


# import pathos.multiprocessing as multiprocessing

# SchemaUpdate(credentials, logger, chunks[0])
# data_group = f[data_group_name]['data']
# results = None

# num_stored = 0
# pool = multiprocessing.ProcessPool(multiprocessing.cpu_count())
# results = pool.uimap(download_chunk, chunks)
# for num_rows, chunk in results:
#     num_stored += 1
#     print(f'Stored {num_rows}, chunk {num_stored} / {len(chunks)}.')
#     # writer.write_batch(record_batch)
# pool.close()
# pool.join()

# print('Done.')

if __name__ == "__main__":
    main()
