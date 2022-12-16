import io
from itertools import product
import json
from multiprocessing import Pool
import os
import uuid
from psycopg2 import sql

import pyarrow
import pyarrow.parquet as parquet
from jobqueue import *
from jobqueue.cursor_manager import CursorManager
import numpy
from sqlalchemy import column

from dmp.logging.postgres_parameter_map import PostgresParameterMap
import sys


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
    col_map = {name : i for i, name in enumerate(columns)}

    with CursorManager(credentials, name=str(uuid.uuid1()), autocommit=False) as cursor:
        cursor.itersize = 8

        q = sql.SQL('SELECT ')
        q += sql.SQL(', ').join([
            sql.SQL('{} {}').format(sql.Identifier(c), sql.Identifier(c))
            for c in columns
        ])

        q += sql.SQL(' FROM run_ r ')
        q += sql.SQL(' ORDER BY record_timestamp DESC LIMIT 1000')
        q += sql.SQL(' ;')

        x = cursor.mogrify(q)
        print(x)
        cursor.execute(q)
        shrinkage = []
        for row in cursor:

            result_block = {}
            for name in history_columns:
                v = row[col_map[name]]
                if v is not None and len(v) > 0:
                    result_block[name] = v
            
            if len(result_block) > 0:
                num_epochs = max((len(h) 
                    for h in result_block.values()
                    if h is not None
                    ))
                for h in result_block.values():
                    h.extend([float('NaN')] * (num_epochs - len(h)))
                
                result_block['epoch'] = list(range(1,1+num_epochs))

                uncompressed : int = 0
                use_byte_stream_split = []
                fields = []
                max_relative_error = 0.0
                for name, h in list(result_block.items()):
                    e = h[0]
                    ha = numpy.array(h, dtype=type(e))

                    ptype = None
                    ntype = None
                    hi = numpy.max(h)
                    lo = numpy.min(h)

                    if isinstance(e, int):
                        if hi <= (2**15-1) and lo >= (-2**15):
                            ptype = pyarrow.int16()
                            ntype = numpy.int16                            
                        elif hi <= (2**31-1) and lo >= (-2**31):
                            ptype = pyarrow.int32()
                            ntype = numpy.int32
                        ptype = pyarrow.int64()
                        ntype = numpy.int64
                    elif isinstance(e, float):
                        ptype = pyarrow.float32()
                        ntype = numpy.float32
                        # ha = numpy.array(ha, dtype=numpy.float16)
                        significand, exponent = numpy.frexp(ha)
                        significand = numpy.array(significand, dtype=numpy.float16)
                        significand = numpy.array(significand, dtype=numpy.float32)
                        ha = numpy.ldexp(significand, exponent)
                        uncompressed += 4
                        use_byte_stream_split.append(name)

                        h_orig = numpy.array(h)
                        error = numpy.abs(ha - h_orig)
                        error *= 1-numpy.isnan(h)
                        relative = error / (1e-32 * (h_orig == 0.0) + numpy.abs(h_orig))
                        # relative = relative * (numpy.abs(h_orig) >= 1e-6)
                        max_relative_error = max(max_relative_error, numpy.max(relative))
                        # print(f'{name} error: {round(numpy.max(error)*100, 6)}, {round(numpy.max(relative)*100, 6)}%')

                    else:
                        raise NotImplementedError(f'Unhandled type {type(e)} for {name}.')
                    fields.append(pyarrow.field(name, ptype))
                    array_version = numpy.array(ha, dtype=ntype)
                    result_block[name] = array_version

                   
                uncompressed *= num_epochs

                schema = pyarrow.schema(fields)
                table = pyarrow.Table.from_pydict(result_block, schema=schema)
                
                buffer = io.BytesIO()
                pyarrow_file = pyarrow.PythonFile(buffer)

                parquet.write_table(
                        table,
                        pyarrow_file,
                        # root_path=dataset_path,
                        # schema=schema,
                        # partition_cols=partition_cols,
                        # data_page_size=128 * 1024,
                        compression='ZSTD',
                        # compression='BROTLI',
                        compression_level=19,
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
                print(f'Written {bytes_written} bytes, {uncompressed} uncompressed, ratio {uncompressed/bytes_written}, shrinkage {bytes_written/uncompressed}, max relative error {round(numpy.max(max_relative_error)*100, 6)}%.')
                shrinkage.append(bytes_written/uncompressed)

        shrinkage = numpy.array(shrinkage)
        print(f'average shrinkage: {numpy.average(shrinkage)} average ratio: {numpy.average(1.0/shrinkage)}')
    

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
