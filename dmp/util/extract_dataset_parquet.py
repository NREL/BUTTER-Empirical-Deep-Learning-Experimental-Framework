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


def main():
    credentials = connect.load_credentials('dmp')
    parameter_map = None
    with CursorManager(credentials) as cursor:
        parameter_map = PostgresParameterMap(cursor)

    # base_path = '/projects/dmpapps/jperrsau/datasets/2022_05_20_fixed_3k_1/'
    # base_path = '/home/ctripp/scratch/'
    dataset_path = '../experiment_summary/'
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
        # pyarrow.field('python_version', pyarrow.string(), nullable=True),
        # pyarrow.field('run_config.shuffle', pyarrow.bool_(), nullable=True),

        pyarrow.field('task', pyarrow.string(), nullable=True),
        # pyarrow.field('task_version', pyarrow.uint16(), nullable=True),
        # pyarrow.field('tensorflow_version', pyarrow.string(), nullable=True),
        pyarrow.field('test_split', pyarrow.float32(), nullable=True),
        # pyarrow.field('test_split_method', pyarrow.string(), nullable=True),
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
        pyarrow.field('experiment_id', pyarrow.uint32(), nullable=False),
        
        pyarrow.field('primary_sweep', pyarrow.bool_(), nullable=False),
        pyarrow.field('300_epoch_sweep', pyarrow.bool_(), nullable=False),
        pyarrow.field('30k_epoch_sweep', pyarrow.bool_(), nullable=False),
        pyarrow.field('learning_rate_sweep', pyarrow.bool_(), nullable=False),
        pyarrow.field('label_noise_sweep', pyarrow.bool_(), nullable=False),
        pyarrow.field('batch_size_sweep', pyarrow.bool_(), nullable=False),
        pyarrow.field('regularization_sweep', pyarrow.bool_(), nullable=False),
        
        pyarrow.field('num_free_parameters', pyarrow.uint64()),
        pyarrow.field('widths', pyarrow.list_(pyarrow.uint32())),
        pyarrow.field('network_structure', pyarrow.string()),
        pyarrow.field('num_runs', pyarrow.uint8()),
        pyarrow.field('num', pyarrow.list_(pyarrow.uint8())),
        pyarrow.field('val_loss_num_finite', pyarrow.list_(pyarrow.uint8())),
        pyarrow.field('val_loss_avg', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_loss_stddev', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_loss_min', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_loss_max', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('loss_num_finite', pyarrow.list_(pyarrow.uint8())),
        pyarrow.field('loss_avg', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('loss_stddev', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('loss_min', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('loss_max', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('loss_median', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_accuracy_avg', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_accuracy_stddev', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_accuracy_median', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_loss_median', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('accuracy_avg', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('accuracy_stddev', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('accuracy_median', pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_mean_squared_error_avg',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_mean_squared_error_stddev',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_mean_squared_error_median',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('mean_squared_error_avg',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('mean_squared_error_stddev',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('mean_squared_error_median',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_kullback_leibler_divergence_avg',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_kullback_leibler_divergence_stddev',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('val_kullback_leibler_divergence_median',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('kullback_leibler_divergence_avg',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('kullback_leibler_divergence_stddev',
                      pyarrow.list_(pyarrow.float32())),
        pyarrow.field('kullback_leibler_divergence_median',
                      pyarrow.list_(pyarrow.float32())),
    ]

    column_name_mapping = {
        'val_loss_num_finite': 'train_test_loss_num_finite',
        'val_loss_avg': 'train_test_loss_avg',
        'val_loss_stddev': 'train_test_loss_stddev',
        'val_loss_min': 'train_test_loss_min',
        'val_loss_max': 'train_test_loss_max',
        'loss_num_finite': 'train_loss_num_finite',
        'loss_avg': 'train_loss_avg',
        'loss_stddev': 'train_loss_stddev',
        'loss_min': 'train_loss_min',
        'loss_max': 'train_loss_max',
        'loss_median': 'train_loss_median',
        'val_accuracy_avg': 'train_test_accuracy_avg',
        'val_accuracy_stddev': 'train_test_accuracy_stddev',
        'val_accuracy_median': 'train_test_accuracy_median',
        'val_loss_median': 'train_test_loss_median',
        'accuracy_avg': 'train_accuracy_avg',
        'accuracy_stddev': 'train_accuracy_stddev',
        'accuracy_median': 'train_accuracy_median',
        'val_mean_squared_error_avg': 'train_test_mean_squared_error_avg',
        'val_mean_squared_error_stddev': 'train_test_mean_squared_error_stddev',
        'val_mean_squared_error_median': 'train_test_mean_squared_error_median',
        'mean_squared_error_avg': 'train_mean_squared_error_avg',
        'mean_squared_error_stddev': 'train_mean_squared_error_stddev',
        'mean_squared_error_median': 'train_mean_squared_error_median',
        'val_kullback_leibler_divergence_avg': 'train_test_kullback_leibler_divergence_avg',
        'val_kullback_leibler_divergence_stddev': 'train_test_kullback_leibler_divergence_stddev',
        'val_kullback_leibler_divergence_median': 'train_test_kullback_leibler_divergence_median',
        'kullback_leibler_divergence_avg': 'train_kullback_leibler_divergence_avg',
        'kullback_leibler_divergence_stddev': 'train_kullback_leibler_divergence_stddev',
        'kullback_leibler_divergence_median': 'train_kullback_leibler_divergence_median',
    }

    inverse_column_name_mapping = {v: k for k, v in column_name_mapping.items()}

    for f in data_columns:
        if f.name in column_name_mapping:
            f.name = column_name_mapping[f.name]

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

    chunk_size = 128
    chunks = []
    with CursorManager(credentials) as cursor:

        q = sql.SQL('SELECT experiment_id, ')
        q += sql.SQL(', ').join(
            [sql.SQL('{}.id {}').format(sql.Identifier(p), sql.Identifier(p))
             for p in partition_cols]
        )
        q += sql.SQL(' FROM experiment_summary_ s ')
        q += sql.SQL(' ').join(
            [sql.SQL(' left join parameter_ {} on ({}.kind = {} and s.experiment_parameters @> array[{}.id]) ').format(
                sql.Identifier(p), sql.Identifier(p), sql.Literal(p), sql.Identifier(p)) for p in partition_cols])

        if len(fixed_parameters) > 0:
            q += sql.SQL(' WHERE ')
            q += sql.SQL(' s.experiment_parameters @> array[{}]::smallint[] ').format(
                sql.SQL(' , ').join([sql.Literal(p) for p in parameter_map.to_parameter_ids(fixed_parameters)]))
        q += sql.SQL(' ORDER BY ')
        q += sql.SQL(' , ').join([sql.SQL('{}.id').format(sql.Identifier(p))
                                  for p in partition_cols] + [sql.SQL('experiment_id')])
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

    # chunks = list(product(*[parameter_map.get_all_ids_for_kind(p) for p in partition_cols]))

    # chunk_size = 16
    # chunks = [
    #     list(experiment_ids[chunk_size*chunk:chunk_size*(chunk+1)])
    #     for chunk in range(int(numpy.ceil(len(experiment_ids) / chunk_size)))]

    def download_chunk(chunk):
        print(f'Begin chunk {chunk}.')
        # chunk = sorted(chunk)
        # non_null_params = sorted([c for c in chunk if c is not None])
        # null_kinds = [partition_cols[i]
        #               for i, c in enumerate(chunk) if c is None]

        result_block = {name: [] for name in column_names}
        row_number = 0

        q = sql.SQL('SELECT experiment_parameters, ')

        q += sql.SQL(', ').join([sql.Identifier(inverse_column_name_mapping.get(c,c))
                                for c in data_column_names])

        q += sql.SQL(' FROM experiment_summary_ s ')
        q += sql.SQL(' WHERE s.experiment_id IN ( ')
        q += sql.SQL(', ').join([sql.Literal(eid) for eid in chunk])
        q += sql.SQL(' );')

        # q += sql.SQL(' WHERE s.experiment_parameters @> array[')
        # q += sql.SQL(', ').join([sql.Literal(p)
        #                             for p in non_null_params])
        # q += sql.SQL(']::smallint[] ')

        # if len(null_kinds) > 0:
        #     q += sql.SQL(' AND NOT (s.experiment_parameters && (')
        #     q += sql.SQL(' SELECT array_agg(id) FROM (')
        #     q += sql.SQL(' UNION ALL ').join([
        #         sql.SQL(' SELECT id from parameter_ where kind = {} ').
        #         format(sql.Literal(k))
        #         for k in null_kinds])
        #     q += sql.SQL(') u )) ')

        # if len(fixed_parameters) > 0:
        #     q += sql.SQL(' AND s.experiment_parameters @> array[{}]::smallint[] ').format(
        #         sql.SQL(' , ').join([sql.Literal(p) for p in parameter_map.to_parameter_ids(fixed_parameters)]))
        # q += sql.SQL(';')

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
                compression_level=9,
                use_dictionary=use_dictionary,
                use_byte_stream_split=use_byte_stream_split,
                data_page_version='2.0',
                existing_data_behavior='overwrite_or_ignore',
                use_legacy_dataset=False,
                # write_batch_size=64,
                # dictionary_pagesize_limit=64*1024,
            )
        print(f'End chunk {chunk}.')
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
                f'Stored {num_rows} in chunk {chunk}, {num_stored} / {len(chunks)}.')
            # writer.write_batch(record_batch)

    print('Done.')


if __name__ == "__main__":
    main()
