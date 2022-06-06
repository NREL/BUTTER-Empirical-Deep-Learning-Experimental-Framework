from itertools import product
import json
from multiprocessing import Pool
import os
import uuid
from psycopg2 import sql

import pyarrow
import pyarrow.parquet as parquet
from jobqueue import connect
from jobqueue.cursor_manager import CursorManager
import numpy

from dmp.logging.postgres_parameter_map import PostgresParameterMap
import sys

def main():
    sweep = None
    if len(sys.argv) >= 2:
        sweep = str(sys.argv[1])

    credentials = connect.load_credentials('dmp')
    parameter_map = None
    with CursorManager(credentials) as cursor:
        parameter_map = PostgresParameterMap(cursor)

    # base_path = '/projects/dmpapps/jperrsau/datasets/2022_05_20_fixed_3k_1/'
    # base_path = '/home/ctripp/scratch/'
    dataset_path = '../experiment/'
    if sweep is not None:
        dataset_path = f'../{sweep}/'
    # file_name = os.path.join(base_path, 'fixed_3k_1.pq')

    # fixed_3k_1_meta.csv.gz

    fixed_parameters = [
        # ('batch', 'fixed_3k_1'),
    ]

    parameter_columns = [
        pyarrow.field('dataset', pyarrow.string(), nullable=True),
        pyarrow.field('shape', pyarrow.string(), nullable=True),
        pyarrow.field('learning_rate', pyarrow.float32(), nullable=True),
        pyarrow.field('batch_size', pyarrow.uint32(), nullable=True),
        pyarrow.field('kernel_regularizer.type',
                      pyarrow.string(), nullable=True),
        pyarrow.field('label_noise', pyarrow.float32(), nullable=True),
        pyarrow.field('depth', pyarrow.uint8(), nullable=True),
        pyarrow.field('epochs', pyarrow.uint32(), nullable=True),
        pyarrow.field('size', pyarrow.uint64(), nullable=True),
        pyarrow.field('batch', pyarrow.string(), nullable=True),
        pyarrow.field('early_stopping', pyarrow.string(), nullable=True),
        pyarrow.field('activation', pyarrow.string(), nullable=True),
        pyarrow.field('input_activation', pyarrow.string(), nullable=True),
        pyarrow.field('output_activation', pyarrow.string(), nullable=True),
        pyarrow.field('kernel_regularizer', pyarrow.string(), nullable=True),
        pyarrow.field('kernel_regularizer.l1',
                      pyarrow.float32(), nullable=True),
        pyarrow.field('kernel_regularizer.l2',
                      pyarrow.float32(), nullable=True),
        pyarrow.field('optimizer', pyarrow.string(), nullable=True),
        pyarrow.field('task', pyarrow.string(), nullable=True),
        pyarrow.field('test_split', pyarrow.float32(), nullable=True),


        pyarrow.field('python_version', pyarrow.string(), nullable=True),
        
        pyarrow.field('task_version', pyarrow.uint16(), nullable=True),
        pyarrow.field('tensorflow_version', pyarrow.string(), nullable=True),
    ]

    partition_cols = [
        'dataset',
        'learning_rate',
        'batch_size',
        'kernel_regularizer.type',
        'label_noise',
        'epochs',
        'shape',
        'depth',
    ]

    data_columns = [
        pyarrow.field('run_id', pyarrow.string()),
        pyarrow.field('experiment_id', pyarrow.uint32()),

        pyarrow.field('primary_sweep',pyarrow.bool_()),
        pyarrow.field('300_epoch_sweep',pyarrow.bool_()),
        pyarrow.field('30k_epoch_sweep',pyarrow.bool_()),
        pyarrow.field('learning_rate_sweep',pyarrow.bool_()),
        pyarrow.field('label_noise_sweep',pyarrow.bool_()),
        pyarrow.field('batch_size_sweep',pyarrow.bool_()),
        pyarrow.field('regularization_sweep',pyarrow.bool_()),

        pyarrow.field('num_free_parameters', pyarrow.uint64()),
        pyarrow.field('widths', pyarrow.list_(pyarrow.uint32())),
        pyarrow.field('network_structure', pyarrow.string(), nullable=True),

        pyarrow.field('platform', pyarrow.string(), nullable=True),
        pyarrow.field('git_hash', pyarrow.string(), nullable=True),
        pyarrow.field('hostname', pyarrow.string(), nullable=True),
        pyarrow.field('seed', pyarrow.int64(), nullable=True),

        pyarrow.field('start_time', pyarrow.int64(), nullable=True),
        pyarrow.field('update_time', pyarrow.int64(), nullable=True),
        pyarrow.field('command', pyarrow.string(), nullable=True),

        pyarrow.field('val_loss', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('loss', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_accuracy', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('accuracy', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_loss', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_loss', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_accuracy', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_accuracy', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_mean_squared_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_mean_squared_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_mean_absolute_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_mean_absolute_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_root_mean_squared_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_root_mean_squared_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_mean_squared_logarithmic_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_mean_squared_logarithmic_error', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_hinge', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_hinge', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_squared_hinge', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_squared_hinge', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_cosine_similarity', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_cosine_similarity', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('test_kullback_leibler_divergence', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('train_kullback_leibler_divergence', pyarrow.list_(pyarrow.float32())),
    ]

    column_name_mapping = {
        'val_loss' : 'test_loss',
        'loss' : 'train_loss',
        'val_accuracy' : 'test_accuracy',
        'accuracy' : 'train_accuracy',
        'val_mean_squared_error' : 'test_mean_squared_error',
        'mean_squared_error' : 'train_mean_squared_error',
        'val_mean_absolute_error' : 'test_mean_absolute_error',
        'mean_absolute_error' : 'train_mean_absolute_error',
        'val_root_mean_squared_error' : 'test_root_mean_squared_error',
        'root_mean_squared_error' : 'train_root_mean_squared_error',
        'val_mean_squared_logarithmic_error' : 'test_mean_squared_logarithmic_error',
        'mean_squared_logarithmic_error' : 'train_mean_squared_logarithmic_error',
        'val_hinge' : 'test_hinge',
        'hinge' : 'train_hinge',
        'val_squared_hinge' : 'test_squared_hinge',
        'squared_hinge' : 'train_squared_hinge',
        'val_cosine_similarity' : 'test_cosine_similarity',
        'cosine_similarity' : 'train_cosine_similarity',
        'val_kullback_leibler_divergence' : 'test_kullback_leibler_divergence',
        'kullback_leibler_divergence' : 'train_kullback_leibler_divergence',
    }

    inverse_column_name_mapping = {v: k for k, v in column_name_mapping.items()}
            
    parameter_column_names = [f.name for f in parameter_columns]
    parameter_column_names_set = set(parameter_column_names)
    data_column_names = [f.name for f in data_columns]
    column_names = parameter_column_names + data_column_names

    columns = parameter_columns + data_columns
    schema = pyarrow.schema(columns)

    # use_dictionary=parameter_column_names,
    # use_byte_stream_split=data_column_names,
    use_byte_stream_split = [
        c.name for c in columns
        if c.type in [
            pyarrow.float32(), pyarrow.list_(pyarrow.float32()),
            pyarrow.float64(), pyarrow.list_(pyarrow.float64()),
        ]
    ]

    use_dictionary = [c.name for c in parameter_columns]

    # Write metadata-only Parquet file from schema
    os.makedirs(dataset_path)
    parquet.write_metadata(
        schema, dataset_path + '_common_metadata')

    chunk_size = 64
    chunks = []
    with CursorManager(credentials) as cursor:

        q = sql.SQL('SELECT run_id, ')
        q += sql.SQL(', ').join(
            [sql.SQL('{}.id {}').format(sql.Identifier(p), sql.Identifier(p))
             for p in partition_cols]
        )
        q += sql.SQL(' FROM run_ r ')
        q += sql.SQL(' ').join(
            [sql.SQL(' left join parameter_ {} on ({}.kind = {} and r.run_parameters @> array[{}.id]) ').format(
                sql.Identifier(p), sql.Identifier(p), sql.Literal(p), sql.Identifier(p)) for p in partition_cols])

        where = False

        if len(fixed_parameters) > 0:
            where = True
            q += sql.SQL(' WHERE ')

            q += sql.SQL(' r.run_parameters @> array[{}]::smallint[] ').format(
                sql.SQL(' , ').join([sql.Literal(p) for p in parameter_map.to_parameter_ids(fixed_parameters)]))

        if sweep is not None:
            if where:
                q += sql.SQL(' AND ')
            else:
                where = True
                q += sql.SQL(' WHERE ')

            q += sql.SQL('EXISTS (SELECT * FROM experiment_ e WHERE e.experiment_id = r.experiment_id AND e.{}) ').format(sql.Identifier(sweep))

        q += sql.SQL(' ORDER BY ')
        q += sql.SQL(' , ').join([sql.SQL('{}.id').format(sql.Identifier(p))
                                  for p in partition_cols] + [sql.SQL('experiment_id'), sql.SQL('run_id')])
        q += sql.SQL(' ;')

        x = cursor.mogrify(q)
        print(x)
        cursor.execute(q)

        chunk = []
        chunk_partition = None
        for row in cursor:
            partition = [row[i+1] for i in range(len(partition_cols))]
            if len(chunk) >= chunk_size or partition != chunk_partition:
                chunk_partition = partition
                chunk = []
                chunks.append(chunk)

            chunk.append(row[0])

    def download_chunk(chunk):
        # print(f'Begin chunk {chunk}.')

        result_block = {name: [] for name in column_names}
        row_number = 0

        q = sql.SQL('SELECT run_parameters, ')

        q += sql.SQL(', ').join([sql.Identifier(inverse_column_name_mapping.get(c,c))
                                for c in data_column_names])
        q += sql.SQL(' FROM ( ')
        q += sql.SQL(' SELECT r.*, e.num_free_parameters num_free_parameters, e.widths widths, e.network_structure network_structure, EXTRACT(epoch FROM s.start_time) start_time, EXTRACT(epoch FROM s.update_time) update_time, d.command command, ')
        q += sql.SQL(' e."primary_sweep" "primary_sweep", ')
        q += sql.SQL(' e."300_epoch_sweep" "300_epoch_sweep", ')
        q += sql.SQL(' e."30k_epoch_sweep" "30k_epoch_sweep", ')
        q += sql.SQL(' e."learning_rate_sweep" "learning_rate_sweep", ')
        q += sql.SQL(' e."label_noise_sweep" "label_noise_sweep", ')
        q += sql.SQL(' e."batch_size_sweep" "batch_size_sweep", ')
        q += sql.SQL(' e."regularization_sweep" "regularization_sweep" ')

        q += sql.SQL(' FROM run_ r JOIN experiment_ e ON (r.experiment_id = e.experiment_id) ')
        q += sql.SQL(' LEFT JOIN job_status s ON (s.id = r.run_id) ')
        q += sql.SQL(' LEFT JOIN job_data d ON (d.id = r.run_id) ')
        q += sql.SQL(' WHERE r.run_id IN ( ')
        q += sql.SQL(', ').join([sql.Literal(eid) for eid in chunk])

        q += sql.SQL(') ')
        q += sql.SQL(') x ')
        q += sql.SQL(' ;')

        with CursorManager(credentials, name=str(uuid.uuid1()), autocommit=False) as cursor:
            cursor.itersize = 8

            cursor.execute(q)
            # if cursor.description is None:
            #     print(cursor.mogrify(q))
            #     continue

            for row in cursor:
                for name in column_names:
                    result_block[name].append(None)

                for kind, value in parameter_map.parameter_from_id(row[0]):
                    if kind in parameter_column_names_set:
                        result_block[kind][row_number] = value

                for i in range(len(data_column_names)):
                    result_block[data_column_names[i]
                                 ][row_number] = row[i+1]

                row_number += 1

        if row_number > 0:

            result_block['network_structure'] = \
                [json.dumps(js, separators=(',', ':'))
                 for js in result_block['network_structure']]

            result_block['run_id'] = [str(e) for e in result_block['run_id']]

            result_block['command'] = \
                [json.dumps(js, separators=(',', ':'))
                 for js in result_block['command']]

            record_batch = pyarrow.Table.from_pydict(
                result_block,
                schema=schema,
            )

            parquet.write_to_dataset(
                record_batch,
                root_path=dataset_path,
                schema=schema,
                partition_cols=partition_cols,
                # data_page_size=128 * 1024,
                compression='BROTLI',
                compression_level=6,
                use_dictionary=use_dictionary,
                use_byte_stream_split=use_byte_stream_split,
                data_page_version='2.0',
                existing_data_behavior='overwrite_or_ignore',
                use_legacy_dataset=False,
                # write_batch_size=64,
                # dictionary_pagesize_limit=64*1024,
            )
        # print(f'End chunk {chunk}.')
        return row_number, chunk

    # with parquet.ParquetWriter(
    #         file_name,
    #         schema=schema,
    #         use_dictionary=parameter_column_names,
    #         # data_page_size=256 * 1024,
    #         compression='BROTLI',
    #         compression_level=8,
    #         use_byte_stream_split=data_column_names,
    #         data_page_version='2.0',
    #         # write_batch_size=64,
    #         # dictionary_pagesize_limit=64*1024,
    #     ) as writer:
    print(f'Created {len(chunks)} chunks.')

    import pathos.multiprocessing as multiprocessing

    # SchemaUpdate(credentials, logger, chunks[0])
    # data_group = f[data_group_name]['data']
    results = None

    num_stored = 0
    with multiprocessing.ProcessPool(38) as pool:
        results = pool.uimap(download_chunk, chunks)
        for num_rows, chunk in results:
            num_stored += 1
            print(
                f'Stored {num_rows}, chunk {num_stored} / {len(chunks)}.')
            # writer.write_batch(record_batch)

    print('Done.')


if __name__ == "__main__":
    main()
