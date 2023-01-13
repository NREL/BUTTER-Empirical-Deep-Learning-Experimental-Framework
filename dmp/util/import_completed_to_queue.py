from math import ceil
import random
from jobqueue.job_status import JobStatus
import ujson
from psycopg import sql
import psycopg.extras as extras
import psycopg
import jobqueue.connect as connect
from pprint import pprint
import sys
from jobqueue.cursor_manager import CursorManager
from dmp.logging.postgres_result_logger import PostgresResultLogger
from dmp.layer.visitor.network_json_deserializer import NetworkJSONDeserializer

from dmp.task.aspect_test.aspect_test_task import AspectTestTask

from dmp.logging.postgres_attribute_map import PostgresAttributeMap
from dmp.jobqueue_interface import jobqueue_marshal

from jobqueue.job_queue import JobQueue

sys.path.append("../../")

psycopg.extras.register_uuid()


class SchemaUpdate:

    def __init__(self, credentials, logger, ids) -> None:
        with CursorManager(credentials) as cursor:
            self.cursor = cursor
            self.logger = logger

            cursor.execute(sql.SQL("""
    SELECT
        j.uuid,
        j.config,
        j.username,
        j.groupname,
        j.host,
        j.status,
        j.worker,
        j.creation_time,
        j.priority,
        j.start_time,
        j.update_time,
        j.end_time,
        j.depth,
        j.wall_time,
        j.retry_count
    FROM jobqueue j
    WHERE j.uuid IN %s
;"""), (ids,))
            rows = list(cursor.fetchall())
            values = [self.convert_row(row) for row in rows]

            psycopg.extras.execute_values(
                cursor,
                sql.SQL("""
WITH v as (
    SELECT
        queue::smallint,
        status::smallint,
        priority::integer,
        id::uuid,
        start_time::timestamp without time zone,
        update_time::timestamp without time zone,
        worker::uuid,
        error_count::smallint,
        error::text,
        command::jsonb,
        parent::uuid
    FROM
        (VALUES %s) AS t (
            queue,
            status,
            priority,
            id,
            start_time,
            update_time,
            worker,
            error_count,
            error,
            command,
            parent
        )
),
i_data AS (
    INSERT INTO job_data (id, command, parent)
    SELECT id, command, parent FROM v
    ON CONFLICT DO NOTHING
)
INSERT INTO job_status (
    queue,
    status,
    priority,
    id,
    start_time,
    update_time,
    worker,
    error_count,
    error
)
    SELECT 
        queue,
        status,
        priority,
        id,
        start_time,
        update_time,
        worker,
        error_count,
        error
    FROM v
ON CONFLICT DO NOTHING
"""),
                values,
                template=None,
                page_size=128
            )

            # pprint(values)

    # 0 j.uuid,
    # 1 j.config,
    # 2 j.username,
    # 3 j.groupname,
    # 4 j.host,
    # 5 j.status,
    # 6 j.worker,
    # 7 j.creation_time,
    # 8 j.priority,
    # 9 j.start_time,
    # 10 j.update_time,
    # 11 j.end_time,
    # 12 j.depth,
    # 13 j.wall_time,
    # 14 j.retry_count,

    def convert_row(self, row):
        queue = -1

        status = JobStatus.Queued.value
        if row[5] == 'done':
            status = JobStatus.Complete.value

        priority = row[14] * 1000000 + random.randint(0, 100000)
        id_ = row[0]
        start_time = row[9]
        update_time = row[10]
        worker = row[6]
        error_count = row[14]
        error = None
        command = jobqueue_marshal.marshal(self.make_task(row[1]))
        parent = None

        return (
            queue,
            status,
            priority,
            id_,
            start_time,
            update_time,
            worker,
            error_count,
            error,
            command,
            parent,
        )

    def make_task(self, config):
        kwargs = {
            'seed': config['seed'],
            'dataset': config['dataset'],
            'input_activation': config['activation'],
            'activation': config['activation'],
            'optimizer': config['optimizer'],
            'shape': config['topology'],
            'size': config['budget'],
            'depth': config['depth'],
            'validation_split': config['run_config']['validation_split'],
            'validation_split_method': config['validation_split_method'],
            'run_config': config['run_config'],
            'label_noise': config['label_noise'],
            'early_stopping': config.get('early_stopping', None),
            'save_every_epochs': None,
        }

        kwargs['batch'] = 'fixed_3k_1'

        if not isinstance(kwargs['early_stopping'], dict):
            kwargs['early_stopping'] = None

        if config.get('residual_mode', None) == 'full':
            kwargs['shape'] = kwargs['shape'] + '_' + 'residual'

        if kwargs['label_noise'] == 'none':
            kwargs['label_noise'] = 0.0

        return AspectTestTask(**kwargs)


if __name__ == "__main__":
    credentials = connect.load_credentials('dmp')

    import simplejson
    # extras.register_default_json(loads=ujson.loads, globally=True)
    # extras.register_default_jsonb(loads=ujson.loads, globally=True)
    extras.register_default_json(loads=simplejson.loads, globally=True)
    extras.register_default_jsonb(loads=simplejson.loads, globally=True)
    psycopg.extensions.register_adapter(dict, psycopg.extras.Json)

    # parameter_map = None
    logger = PostgresResultLogger(credentials)
    ids = None

    queue = JobQueue(credentials, -1, check_table=True)

    with CursorManager(credentials) as cursor:
        # parameter_map = PostgresParameterMap(cursor)

        cursor.execute(sql.SQL("""
    SELECT 
        j.uuid
    FROM jobqueue j
    WHERE j.groupname = 'fixed_3k_1'
    AND (
        NOT EXISTS (SELECT id from job_status where id = j.uuid) OR
        NOT EXISTS (SELECT id from job_data where id = j.uuid))
;"""))

        ids = [row[0] for row in cursor.fetchall()]

    print(f'loaded {len(ids)}.')

    num_readers = 12
    chunk_size = 1024
    chunks = [
        tuple(ids[c*chunk_size: min(len(ids), (c+1)*chunk_size)])
        for c in range(int(ceil(len(ids)/chunk_size)))]

    from pathos.multiprocessing import Pool

    print(f'created {len(chunks)} chunks')
    # SchemaUpdate(credentials, logger, chunks[0])

    def func(c):
        SchemaUpdate(credentials, logger, c)
        return None

    with Pool(num_readers) as p:
        p.map(func, chunks)
