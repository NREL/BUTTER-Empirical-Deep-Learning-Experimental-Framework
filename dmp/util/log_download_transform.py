import sys

import numpy

sys.path.append("../../")

import pandas as pd
import gc
import time
import os
import json
import sqlalchemy

_database = "dmp"
try:
    filename = os.path.join(os.environ['HOME'], ".jobqueue.json")
    _data = json.loads(open(filename).read())
    _credentials = _data[_database]
    user = _credentials["user"]
except KeyError as e:
    raise Exception("No credetials for {} found in {}".format(_database, filename))
connection_string = 'postgresql://{user}:{password}@{host}:5432/{database}'.format(**_credentials)
# connection_string = 'postgresql+asyncpg://{user}:{password}@{host}:5432/{database}'.format(**_credentials)
del _credentials['table_name']

from pathos.multiprocessing import Pool
import math

from psycopg2.extensions import register_adapter, AsIs


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)


def addapt_numpy_array(numpy_array):
    return AsIs(numpy_array.tolist())


register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int32, addapt_numpy_int32)
register_adapter(numpy.ndarray, addapt_numpy_array)

# async def fetch_as_dataframe(con: asyncpg.Connection, query: str, *args):
#     stmt = await con.prepare(query)
#     columns = [a.name for a in stmt.get_attributes()]
#     data = await stmt.fetch(*args)
#     return pandas.DataFrame(data, columns=columns)


drop_list = [
    'loss', 'task', 'endpoint', 'run_name', 'config.run_name', 'val_loss', 'iterations',
    'num_inputs', 'num_classes', 'num_outputs', 'num_weights',
    'num_features', 'num_observations', 'config.log', 'config.rep',
    'config.mode', 'config.name', 'config.reps', 'config.seed',
    'config.depths', 'config.widths',
    'config.budgets', 'config.datasets',
    'config.jq_module',
    'config.run_config.shuffle', 'config.run_config.verbose',
    'config.topologies', 'config.epoch_scale.b', 'config.epoch_scale.m',
    'config.label_noises',
    'config.early_stopping', 'config.learning_rates',
    'config.residual_modes', 'config.network_structure',
    'config.validation_split_method',
    'environment.git_hash',
    'environment.hostname', 'environment.platform',
    'environment.SLURM_JOB_ID', 'environment.python_version',
    'environment.tensorflow_version',
    'config.num_hidden'
]

rename_map = {
    s: s[len('history.'):] for s in [
        'history.loss', 'history.hinge',
        'history.accuracy', 'history.val_loss', 'history.val_hinge',
        'history.val_accuracy', 'history.squared_hinge',
        'history.cosine_similarity', 'history.val_squared_hinge',
        'history.mean_squared_error', 'history.mean_absolute_error',
        'history.val_cosine_similarity', 'history.val_mean_squared_error',
        'history.root_mean_squared_error', 'history.val_mean_absolute_error',
        'history.kullback_leibler_divergence',
        'history.val_root_mean_squared_error',
        'history.mean_squared_logarithmic_error',
        'history.val_kullback_leibler_divergence',
        'history.val_mean_squared_logarithmic_error',
    ]}

rename_map.update({
    s: s[len('config.'):] for s in
    [
        'config.mode', 'config.name', 'config.reps', 'config.seed',
        'config.depth', 'config.budget', 'config.depths', 'config.widths',
        'config.budgets', 'config.dataset', 'config.datasets',
        'config.run_name', 'config.topology', 'config.jq_module',

        'config.activation', 'config.num_hidden', 'config.run_config.epochs',
        'config.run_config.shuffle', 'config.run_config.verbose',
        'config.run_config.batch_size', 'config.run_config.validation_split',
        'config.topologies', 'config.epoch_scale.b', 'config.epoch_scale.m',
        'config.label_noise', 'config.label_noises', 'config.residual_mode',
        'config.early_stopping', 'config.learning_rates',
        'config.residual_modes', 'config.network_structure',
        'config.validation_split_method',
    ]
})

rename_map.update({
    'config.optimizer.config.learning_rate': 'learning_rate',
    'config.optimizer.class_name': 'optimizer',
    'config.run_config.epochs': 'epochs',
    'config.run_config.batch_size': 'batch_size',
    'config.run_config.validation_split': 'validation_split'

})
print(rename_map)

for k in drop_list:
    rename_map.pop(k, None)

type_map = {
    'depth': 'int16',
    'budget': 'int64',
    'dataset': 'str',
    'topology': 'str',
    'learning_rate': 'float32',
    'optimizer': 'str',
    'activation': 'str',
    'epochs': 'int32',
    'batch_size': 'int32',
    'validation_split': 'float32',
    # 'label_noise' : 'float32',
    'residual_mode': 'str',
    'job_length': 'str',
    # 'loss' : 'float32[]',
    # 'hinge' : 'float32[]',
    # 'accuracy' : 'float32[]',
    # 'val_loss' : 'float32[]',
    # 'val_hinge' : 'float32[]',
    # 'val_accuracy' : 'float32[]',
    # 'squared_hinge' : 'float32[]',
    # 'cosine_similarity' : 'float32[]',
    # 'val_squared_hinge' : 'float32[]',
    # 'mean_squared_error' : 'float32[]',
    # 'mean_absolute_error' : 'float32[]',
    # 'val_cosine_similarity' : 'float32[]',
    # 'val_mean_squared_error' : 'float32[]',
    # 'root_mean_squared_error' : 'float32[]',
    # 'val_mean_absolute_error' : 'float32[]',
    # 'kullback_leibler_divergence' : 'float32[]',
    # 'val_root_mean_squared_error' : 'float32[]',
    # 'mean_squared_logarithmic_error' : 'float32[]',
    # 'val_kullback_leibler_divergence' : 'float32[]',
    # 'val_mean_squared_logarithmic_error' : 'float32[]',
    'id': 'int32',
    'job': 'str',
}

array_cols = [
    'loss',
    'hinge',
    'accuracy',
    'val_loss',
    'val_hinge',
    'val_accuracy',
    'squared_hinge',
    'cosine_similarity',
    'val_squared_hinge',
    'mean_squared_error',
    'mean_absolute_error',
    'val_cosine_similarity',
    'val_mean_squared_error',
    'root_mean_squared_error',
    'val_mean_absolute_error',
    'kullback_leibler_divergence',
    'val_root_mean_squared_error',
    'mean_squared_logarithmic_error',
    'val_kullback_leibler_divergence',
    'val_mean_squared_logarithmic_error',
]


def postprocess_dataframe(data_log):
    # print(f'{len(data_log)} records retrieved. Parsing json...')
    datasets = pd.json_normalize(data_log["doc"])
    data_log.drop(columns=['doc'], inplace=True)
    #     datasets = pd.json_normalize(data_log['doc'].map(orjson.loads))
    # print(datasets.columns)
    datasets.drop(columns=drop_list, inplace=True)
    # print(datasets.columns)
    datasets.rename(columns=rename_map, inplace=True)
    # print(datasets.columns)
    # print(f'Joining with original dataframe...')
    datasets = datasets.join(data_log)
    # print(datasets.columns)
    # print(datasets.dtypes)
    # print(f'converting...')
    datasets = datasets.astype(type_map)

    def convert_label_noise(v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except ValueError:
            return 0.0

    datasets['label_noise'] = datasets['label_noise'].apply(convert_label_noise)

    #
    # datasets['id'] = datasets.astype({'id': 'str'}).dtypes
    # datasets['job'] = datasets.astype({'job': 'str'}).dtypes

    # for col in array_cols:
    #     datasets[col] = datasets[col].apply(lambda e: numpy.array(e, dtype=numpy.single))

    # print(datasets.dtypes)
    return datasets


def func():
    # log_filename = 'aspect_analysis_datasets.feather'
    log_filename = 'fixed_3k_1.parquet'
    groupname = 'fixed_3k_1'
    # engine, session = log._connect()

    #     datasets = None
    #     min_time =  datetime.datetime.fromisoformat('2000-01-01 00:00:00')
    #     try:
    #     #     datasets = pd.read_feather(log_filename)
    #         datasets = pd.read_parquet(log_filename)
    #         min_time = datasets['timestamp'].min()
    #     except:
    #         print(f'Error reading dataset file, {log_filename}.')

    conditions = f'log.groupname = \'{groupname}\''
    q = f'select count(*) from "log" where {conditions}'

    db = sqlalchemy.create_engine(connection_string)
    engine = db.connect()
    count = db.engine.execute(q).scalar()
    engine.close()

    loaded = 0
    chunk_size = 32
    # max_buffered_chunks = 4
    print(f'Loading {count} records from database...')

    # num_readers = min(12, int(math.ceil(count / chunk_size)))
    # num_readers = min(57, int(math.ceil((count/chunk_size))))
    num_readers = min(48, int(math.ceil((count / chunk_size))))
    read_size = int(math.ceil(count / num_readers))
    chunks_per_reader = int(math.ceil(read_size / chunk_size))

    print(f'Reading {count} records with read_size {read_size} and {num_readers} readers...')

    def read_chunk(read_number):
        print(f'Read #{read_number}: starting...')
        db = sqlalchemy.create_engine(connection_string)
        engine = db.connect()
        print(f'Read #{read_number}: connected')

        for i in range(chunks_per_reader):
            q = f'''
            SELECT
            log.id as id,
            log.job as job,
            log.timestamp as timestamp,
            log.groupname as groupname,
            (jobqueue.end_time - jobqueue.start_time) AS job_length,
            log.doc as doc
            FROM
                 log,
                 jobqueue
            WHERE
                {conditions} AND
                jobqueue.uuid = log.job
            ORDER BY id ASC
            LIMIT {chunk_size} OFFSET {read_size * read_number + chunk_size * i}
            '''
            chunk = pd.read_sql(
                    q,
                    engine.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False,
                    params=())

            num_entries = len(chunk)
            print(
                f'Read #{read_number}: processing chunk {i} / {chunks_per_reader} size {num_entries} from database...')
            if num_entries == 0:
                break
            chunk = postprocess_dataframe(chunk)
            print(f'Read #{read_number}: writing chunk to database...')
            chunk.to_sql('materialized_experiments_2', engine, method='multi', if_exists='append', index=False)
            print(f'Read #{read_number}: done writing...')
            # chunks.append(chunk)
            # if len(chunks) >= max_buffered_chunks:
            #     print(f'Read #{read_number}: concatenating {len(chunks)} chunks...')
            #     chunks = [pd.concat(chunks)]
            #     gc.collect()


        # data_generator = pd.read_sql(
        #     q,
        #     engine.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False, chunksize=chunk_size,
        #     params=())
        #
        # print(f'Read #{read_number}: begin read')
        # # chunks = []
        # for i, chunk in enumerate(data_generator):
        #     num_entries = len(chunk)
        #     print(
        #         f'Read #{read_number}: processing chunk {i} / {chunks_per_reader} size {num_entries} from database...')
        #     chunk = postprocess_dataframe(chunk)
        #     print(f'Read #{read_number}: writing chunk to database...')
        #     chunk.to_sql('materialized_experiments_2', engine, method='multi', if_exists='append', index=False)
        #     print(f'Read #{read_number}: done writing...')
        #     # chunks.append(chunk)
        #     # if len(chunks) >= max_buffered_chunks:
        #     #     print(f'Read #{read_number}: concatenating {len(chunks)} chunks...')
        #     #     chunks = [pd.concat(chunks)]
        #     #     gc.collect()

        # results = pd.concat(chunks)
        print(f'Read #{read_number}: complete.')
        return None

    print(f'Initializing pool...')
    start_time = time.perf_counter()
    # for i in range(0, num_readers):
    #     read_chunk(i)
    with Pool(num_readers) as p:
        data_load = p.map(read_chunk, range(0, num_readers))
    # data_load = Parallel(n_jobs=num_readers, batch_size=1, backend='multiprocessing')(
    #     delayed(read_chunk(i)) for i in range(num_readers))

    gc.collect()
    datasets = pd.concat(data_load)
    delta_t = time.perf_counter() - start_time
    num_entries = len(datasets)
    print(f'Processed {num_entries} entries in {delta_t}s at a rate of {num_entries / delta_t} entries / second.')

    if loaded > 0:
        print(f'Loaded {len(datasets)} entries from the database. Saving dataframe to disk...')
        datasets.to_parquet(log_filename)
        print(f'Write complete.')

    #     print(f'A total of {len(datasets)} records loaded.')

    data_log = None
    additional_datasets = None
    additional_data_log = None
    # log._close(engine, session)

    gc.collect()


func()
