import sys

import numpy

sys.path.append("../../")

import pandas as pd
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
    'config.num_hidden',
    'config.test_split'
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

canonical_cols = [
    'groupname',
    'dataset',
    'topology',
    'residual_mode',
    'optimizer',
    'activation',
]

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

base_cols = [
    'id',
    'job',
    'timestamp',
    'budget',
    'depth',
    'learning_rate',
    'epochs',
    'batch_size',
    'validation_split',
    'label_noise',
    'groupname',
    'dataset',
    'topology',
    'residual_mode',
    'optimizer',
    'activation',
    'job_length',
]

history_cols = ['id']
history_cols.extend(array_cols)

dest_cols = set(base_cols)
dest_cols.update(history_cols)

string_map = {}


def postprocess_dataframe(data_log, engine):
    # print(f'{len(data_log)} records retrieved. Parsing json...')
    datasets = pd.json_normalize(data_log["doc"])
    data_log.drop(columns=['doc'], inplace=True)
    #     datasets = pd.json_normalize(data_log['doc'].map(orjson.loads))
    # print(datasets.columns)

    if 'config.validation_split_method' not in datasets.columns:
        datasets['config.validation_split_method'] = 'old'
    datasets['config.validation_split_method'].fillna(value='old', inplace=True)

    if 'config.label_noise' not in datasets.columns:
        datasets['config.label_noise'] = 0.0
    datasets['config.label_noise'].fillna(value=0.0, inplace=True)

    # print(datasets.columns)
    col_set = set(datasets.columns)
    datasets.drop(columns=[c for c in drop_list if c in col_set], inplace=True)

    # print(datasets.columns)
    col_set = set(datasets.columns)
    datasets.rename(columns={k: v for k, v in rename_map.items() if k in col_set}, inplace=True)
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
    # print(datasets.columns)


    def canonicalize_string(s):
        if s in string_map:
            return string_map[s]
        print(f'String miss on {s}.')
        engine.execute('INSERT INTO strings(value) VALUES(%s) ON CONFLICT(value) DO NOTHING', (s,))
        str_id = engine.execute('SELECT id from strings WHERE value = %s', (s, )).scalar()
        string_map[s] = str_id
        return str_id

    for col in canonical_cols:
        datasets[col] = datasets[col].apply(canonicalize_string)

    datasets.drop(columns=[c for c in datasets.columns if c not in dest_cols], inplace=True)
    base = datasets.filter(base_cols, axis=1)
    history = datasets.filter(history_cols, axis=1)
    print(base)
    # print(base.columns)
    # print(history.columns)
    # print(datasets.columns)
    #
    # datasets['id'] = datasets.astype({'id': 'str'}).dtypes
    # datasets['job'] = datasets.astype({'job': 'str'}).dtypes

    # for col in array_cols:
    #     datasets[col] = datasets[col].apply(lambda e: numpy.array(e, dtype=numpy.single))

    # print(datasets.dtypes)
    return base, history


from sqlalchemy.dialects.postgresql import insert


def insert_on_duplicate(table, conn, keys, data_iter):
    insert_stmt = insert(table.table).values(list(data_iter))
    insert_stmt.on_conflict_do_nothing()
    conn.execute(insert_stmt)


def func():
    # log_filename = 'aspect_analysis_datasets.feather'
    log_filename = 'fixed_3k_1.parquet'
    groupnames = ('fixed_3k_1', 'fixed_3k_0')
    source_table = 'log'
    dest_table_base = 'materialized_experiments_3_base'
    dest_table_history = 'materialized_experiments_3_history'
    num_threads = 64
    # num_threads = 1
    # engine, session = log._connect()

    #     datasets = None
    #     min_time =  datetime.datetime.fromisoformat('2000-01-01 00:00:00')
    #     try:
    #     #     datasets = pd.read_feather(log_filename)
    #         datasets = pd.read_parquet(log_filename)
    #         min_time = datasets['timestamp'].min()
    #     except:
    #         print(f'Error reading dataset file, {log_filename}.')

    db = sqlalchemy.create_engine(connection_string)
    engine = db.connect()

    string_map_df =  pd.read_sql(
        f'''SELECT id, value from strings''',
        engine.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False,
        params=())

    values = string_map_df['value'].to_list()
    for i, str_id in enumerate(string_map_df['id'].to_list()):
        string_map[values[i]] = str_id

    conditions = f'log.groupname IN {groupnames}'
    q = f'''
    select log.id from {source_table} AS log 
    where {conditions} AND NOT EXISTS (SELECT id FROM {dest_table_base} AS d WHERE d.id = log.id)'''


    # count = db.engine.execute(q).scalar()
    ids = pd.read_sql(
        q,
        engine.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False,
        params=())
    engine.close()

    ids = ids['id'].to_numpy()
    count = ids.size

    loaded = 0
    chunk_size = 16
    # max_buffered_chunks = 4
    print(f'Loading {count} records from database...')

    # num_readers = min(12, int(math.ceil(count / chunk_size)))
    # num_readers = min(57, int(math.ceil((count/chunk_size))))
    num_readers = min(num_threads, int(math.ceil((count / chunk_size))))
    read_size = int(math.ceil(count / num_readers))

    print(f'Reading {count} records with read_size {read_size} and {num_readers} readers...')

    def read_chunk(read_number):
        print(f'Read #{read_number}: starting...')
        db = sqlalchemy.create_engine(connection_string)
        engine = db.connect()
        print(f'Read #{read_number}: connected')

        start_index = read_number * read_size
        end_index = min(start_index + read_size, count)
        num_to_read = end_index - start_index
        num_chunks = int(math.ceil(num_to_read / chunk_size))
        for i in range(num_chunks):
            from_index = start_index + i * chunk_size
            to_index = from_index + chunk_size
            ids_for_this_chunk = tuple(ids[from_index:to_index])
            q = f'''
            SELECT
            log.id as id,
            log.job as job,
            log.timestamp as timestamp,
            log.groupname as groupname,
            (jobqueue.end_time - jobqueue.start_time) AS job_length,
            log.doc as doc
            FROM
                 {source_table} AS log,
                 jobqueue
            WHERE
                jobqueue.uuid = log.job AND
                log.id IN {ids_for_this_chunk}
            '''
            chunk = pd.read_sql(
                q,
                engine.execution_options(stream_results=True, postgresql_with_hold=True), coerce_float=False,
                params=())

            num_entries = len(chunk)
            print(
                f'Read #{read_number}: processing chunk {i} / {num_chunks} size {num_entries} from database...')
            if num_entries == 0:
                break
            base_chunk, history_chunk = postprocess_dataframe(chunk, engine)
            print(f'Read #{read_number}: writing chunk to database...')
            base_chunk.to_sql(dest_table_base, engine, method=insert_on_duplicate, if_exists='append', index=False)
            history_chunk.to_sql(dest_table_history, engine, method=insert_on_duplicate, if_exists='append',
                                 index=False)
            print(f'Read #{read_number}: done writing...')
            del base_chunk
            del history_chunk
            del chunk

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

    delta_t = time.perf_counter() - start_time
    print(f'Processed {count} entries in {delta_t}s at a rate of {count / delta_t} entries / second.')


func()
