from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas
import io
from itertools import product
import json
from multiprocessing import Pool
import os
from pprint import pprint
import uuid
from psycopg import sql

import pyarrow
import pyarrow.parquet as parquet
from jobqueue import load_credentials
from jobqueue.cursor_manager import CursorManager
import numpy
from sqlalchemy import column
from tensorflow import shape
from dmp.dataset.dataset_spec import DatasetSpec
from dmp.dataset.ml_task import MLTask
from dmp.dataset.prepared_dataset import PreparedDataset
from dmp.layer.dense import Dense
from dmp.logging.postgres_compressed_result_logger import PostgresCompressedResultLogger

from dmp.postgres_interface.postgres_attr_map import PostgresAttrMap
import sys
from dmp.logging.postgres_parameter_map_v1 import PostgresParameterMapV1
from dmp.model.dense_by_size import DenseBySize

from dmp.parquet_util import make_pyarrow_schema
from dmp.task.experiment.training_experiment.training_experiment import TrainingExperiment

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


def main():

    credentials = load_credentials('dmp')
    old_parameter_map = None
    new_parameter_map = None
    with CursorManager(credentials) as cursor:
        old_parameter_map = PostgresParameterMapV1(cursor)
        new_parameter_map = PostgresAttrMap(cursor)

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

    experiment_columns = [
        'experiment_id',
        'num_free_parameters',
        'network_structure',
        'widths',
        'size',
        'relative_size_error',
        'primary_sweep',
        '300_epoch_sweep',
        '30k_epoch_sweep',
        'learning_rate_sweep',
        'label_noise_sweep',
        'batch_size_sweep',
        'regularization_sweep',
        'optimizer_sweep',
        'learning_rate_batch_size_sweep',
        'size_adjusted_regularization_sweep',
        'butter',
    ]

    run_columns = [
        'run_id',
        'job_id',
        'run_parameters',
        'record_timestamp',
        'platform',
        'git_hash',
        'hostname',
        'slurm_job_id',
        'seed',
        'save_every_epochs',
        'num_gpus',
        'num_nodes',
        'num_cpus',
        'gpu_memory',
        'nodes',
        'cpus',
        'gpus',
        'strategy',
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

    columns = experiment_columns + run_columns + history_columns

    column_index_map = {name: i for i, name in enumerate(columns)}

    result_logger = PostgresCompressedResultLogger(credentials)

    with CursorManager(credentials, name=str(uuid.uuid1()),
                       autocommit=False) as cursor:
        cursor.itersize = 8

        q = sql.SQL('SELECT ')
        q += sql.SQL(', ').join([
            sql.SQL('e.{col} {col}').format(col=sql.Identifier(c))
            for c in experiment_columns
        ] + [
            sql.SQL('r.{col} {col}').format(col=sql.Identifier(c))
            for c in (run_columns + history_columns)
        ])

        q += sql.SQL(' FROM run_ r INNER JOIN experiment_ e')
        q += sql.SQL(' ON r.experiment_id = e.experiment_id')
        q += sql.SQL(' ORDER BY r.record_timestamp DESC LIMIT 10')
        q += sql.SQL(';')

        x = cursor.mogrify(q)
        print(x)
        cursor.execute(q)
        shrinkage = []
        for row in cursor:

            def get_cell(column: str):
                return row[column_index_map[column]]

            src_parameters = {
                kind: value
                for kind, value in old_parameter_map.parameter_from_id(
                    get_cell('run_parameters'), cursor=cursor)
            }

            if not src_parameters.get('run_config.shuffle', False):
                continue

            if src_parameters.get('task', None) != 'AspectTestTask':
                continue

            if src_parameters.get('early_stopping', None) is not None:
                continue

            if src_parameters.get('input_activation', 'relu') != 'relu':
                continue

            dataset_src = 'pmlb'
            dataset_name = src_parameters['dataset']
            dsinfo = dataset_index[dataset_index['Dataset'] ==
                                   dataset_name].iloc[0]

            ml_task = MLTask.regression
            num_outputs = 1
            if dsinfo['Task'] == 'classification':
                ml_task = MLTask.classification
                num_outputs = int(dsinfo['n_classes'])
                if num_outputs == 2:
                    num_outputs = 1
            dataset_size = int(dsinfo['n_observations'])
            test_split = float(src_parameters['test_split'])
            train_size = math.floor(dataset_size * test_split)

            optimizer_class = src_parameters.get('optimizer', 'Adam')
            if optimizer_class == 'adam':
                optimizer_class = 'Adam'
            optimizer = {
                'class': optimizer_class,
                'learning_rate':
                float(src_parameters.get('learning_rate', 0.0001))
            }
            momentum = src_parameters.get('optimizer.config.momentum', None)
            if momentum is not None:
                optimizer['momentum'] = float(momentum)
            nesterov = src_parameters.get('optimizer.config.nesterov', None)
            if nesterov is not None:
                optimizer['nesterov'] = nesterov

            # activity_regularizer
            def make_keras_config(prefix: str) -> Optional[Dict[str, Any]]:
                _class = src_parameters.get(prefix + '.type', None)
                if _class is None:
                    _class = src_parameters.get(prefix, None)
                    if _class is None:
                        return None

                activity_regularizer = {'class': _class}
                for k, v in src_parameters.items():
                    if k.startswith(prefix) and not k.endswith('.type'):
                        activity_regularizer[k[len(prefix) + 1:]] = v
                return activity_regularizer

            layer_config: Dict[str, Any] = {
                'kernel_initializer': 'GlorotUniform',
            }

            layer_config.update({
                k: make_keras_config(k)
                for k in [
                    'kernel_regularizer',
                    'bias_regularizer',
                    'activity_regularizer',
                ]
            })

            shape = str(src_parameters['shape'])

            # input shape, output shape, ml_task
            experiment = TrainingExperiment(
                seed=int(get_cell('seed')),
                batch=str(src_parameters.get('batch', None)),  # type: ignore
                dataset=DatasetSpec(
                    name=dataset_name,
                    source=dataset_src,
                    method=src_parameters.get('test_split_method',
                                              'shuffled_train_test_split'),
                    test_split=float(src_parameters.get('test_split', 0.2)),
                    validation_split=0.0,
                    label_noise=float(src_parameters.get('label_noise', 0.0)),
                ),
                model=DenseBySize(
                    input=None,
                    output=Dense.make(
                        num_outputs,
                        layer_config | {
                            'activation':
                            src_parameters.get('output_activation', None),
                        },
                    ),
                    shape=shape,
                    size=int(src_parameters['size']),
                    depth=int(src_parameters['depth']),
                    search_method='integer',
                    inner=Dense.make(
                        -1,
                        layer_config | {
                            'activation': src_parameters.get(
                                'activation', 'relu'),
                        },
                    ),
                ),
                fit={
                    'batch_size': int(src_parameters['batch_size']),
                    'epochs': int(src_parameters['epochs']),
                },
                optimizer=optimizer,
                loss=None,
                early_stopping=None,
                record_post_training_metrics=False,
                record_times=False,
                record_model=None,
                record_metrics=None,
            )

            def get_input_shape(target) -> List[int]:
                if target[''] == 'NInput':
                    return target['shape']
                return get_input_shape(target['inputs'][0])

            prepared_dataset = PsuedoPreparedDataset(
                ml_task=ml_task,
                # input_shape=[int(dsinfo['n_features'])],
                input_shape=get_input_shape(get_cell('network_structure')),
                output_shape=[num_outputs],
                train_size=train_size,
                test_size=dataset_size - train_size,
                validation_size=0,
            )

            # prepared_dataset = PreparedDataset(
            #     experiment.dataset,
            #     int(src_parameters['batch_size']),
            # )

            metrics = experiment._autoconfigure_for_dataset(
                prepared_dataset, )  # type: ignore
            metric_names = [
                m if isinstance(m, str) else m.name for m in metrics
            ]
            metric_names.append('loss')
            print(metric_names)

            network = experiment._make_network(experiment.model)
            pprint(get_cell('widths'))
            pprint(get_cell('network_structure'))
            if network.num_free_parameters != get_cell('num_free_parameters'):
                if shape.startswith('wide_first') and shape != :
                    # rety with wide first 2x
                    
                print(
                    f'num_free_parameters {network.num_free_parameters} != {get_cell("num_free_parameters")}'
                )
                pprint(experiment)
                pprint(dsinfo)
                pprint(get_cell('widths'))
                pprint(get_cell('network_structure'))
                pprint(marshal.marshal(network))
                continue

            def map_resource_list(src: Optional[str],) ->  Tuple[Optional[List[int]], Optional[int]]:
                if not isinstance(src, str) or len(src) < 2:
                    return None, None
                l = [int(i) for i in (src[1:-1].split(',')) if len(i) > 0]
                return l, len(l)

            history = {}
            for m in metric_names:
                for p in ['test_', 'train_']:
                    k = p + m
                    try:
                        history[k] = get_cell(k)
                    except KeyError:
                        continue
            num_epochs = max(
                (len(h) for h in history.values() if h is not None))
            for h in history.values():
                h.extend([None] * (num_epochs - len(h)))
            history['epoch'] = list(range(1, num_epochs + 1))


            worker_info = {
                'gpu_memory': get_cell('gpu_memory'),
                'strategy': get_cell('strategy'),
            }

            for key in ['cpu', 'gpu', 'node']:
                list_key = key + 's'
                l, num = map_resource_list(get_cell(list_key))
                worker_info[list_key] = l
                worker_info['num_' + key] = num

            result_record = experiment._make_result_record(
                worker_info=worker_info,
                job_id=get_cell('job_id'),
                dataset=prepared_dataset,  # type: ignore
                network=network,
                history=history,
            )

            result_record.run_data.update({
                'run_id':
                get_cell('run_id'),
                'python_version':
                src_parameters.get('python_version', None),
                'platform':
                get_cell('platform'),
                'tensorflow_version':
                src_parameters.get('tensorflow_version', None),
                'hostname':
                get_cell('hostname'),
                'slurm_job_id':
                get_cell('slurm_job_id'),
                'git_hash':
                get_cell('git_hash'),
                'task_version':
                int(src_parameters.get('task_version', 0))
            })

            for k in [
                    'primary_sweep',
                    '300_epoch_sweep',
                    '30k_epoch_sweep',
                    'learning_rate_sweep',
                    'label_noise_sweep',
                    'batch_size_sweep',
                    'regularization_sweep',
                    'optimizer_sweep',
                    'learning_rate_batch_size_sweep',
                    'size_adjusted_regularization_sweep',
                    'butter',
            ]:
                if get_cell(k):
                    result_record.experiment_attrs[k] = True

            # pprint(src_parameters)
            # print(dsinfo)
            # # pprint(marshal.marshal(result_record))

            # print('experiment_parameters')
            # pprint(result_record.experiment_parameters)

            # print('experiment_data')
            # pprint(result_record.experiment_data)

            # print('run_data')
            # pprint(result_record.run_data)

            result_logger.log(result_record,
                              experiment_id=get_cell('experiment_id'))

            # if len(history) > 0:
            #     schema, use_byte_stream_split = make_pyarrow_schema(
            #         history.items())

            #     table = pyarrow.Table.from_pydict(history, schema=schema)

            #     buffer = io.BytesIO()
            #     pyarrow_file = pyarrow.PythonFile(buffer)

            #     parquet.write_table(
            #         table,
            #         pyarrow_file,
            #         # root_path=dataset_path,
            #         # schema=schema,
            #         # partition_cols=partition_cols,
            #         data_page_size=8 * 1024,
            #         # compression='BROTLI',
            #         # compression_level=8,
            #         compression='ZSTD',
            #         compression_level=12,
            #         use_dictionary=False,
            #         use_byte_stream_split=use_byte_stream_split,
            #         version='2.6',
            #         data_page_version='2.0',
            #         # existing_data_behavior='overwrite_or_ignore',
            #         # use_legacy_dataset=False,
            #         write_statistics=False,
            #         # write_batch_size=64,
            #         # dictionary_pagesize_limit=64*1024,
            #     )
            #     bytes_written = buffer.getbuffer().nbytes
            #     # print(
            #     #     f'Written {bytes_written} bytes, {uncompressed} uncompressed, ratio {uncompressed/bytes_written}, shrinkage {bytes_written/uncompressed}, max relative error {round(numpy.max(max_relative_error)*100, 6)}%.'
            #     # )
            #     print(f'Written {bytes_written} bytes.')
            #     # shrinkage.append(bytes_written / uncompressed)
            '''
            # TODO: additional run cols
            record_timestamp
            experiment_id

            '''

        # shrinkage = numpy.array(shrinkage)
        # print(
        #     f'average shrinkage: {numpy.average(shrinkage)} average ratio: {numpy.average(1.0/shrinkage)}'
        # )


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
