import sys
import uuid

from jobqueue.job_queue import JobQueue
from jobqueue import load_credentials
from dmp import common
from dmp.postgres_interface.postgres_compressed_result_logger import PostgresCompressedResultLogger
from dmp.postgres_interface.schema.postgres_schema import PostgresSchema
from dmp.worker import Worker

import tensorflow

# from .common import jobqueue_marshal


def make_strategy(num_cores, gpus, gpu_mem):
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    print(
        f'Found: {len(gpus)} GPUs. Using: {gpus}.'
    )
    gpu_set = set(gpus)
    gpu_devices = []
    for gpu in gpus:
        number = int(gpu.name.split(':')[-1])
        if number in gpu_set:
            gpu_devices.append(gpu)
            tensorflow.config.experimental.set_virtual_device_configuration(
                gpu, [
                    tensorflow.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpu_mem)
                ])

    cpus = tensorflow.config.experimental.list_physical_devices('CPU')
    # print(f'cpus: {cpus}')
    visible_devices = gpu_devices.copy()
    visible_devices.extend(cpus)
    tensorflow.config.set_visible_devices(visible_devices)
    tensorflow.config.set_soft_device_placement(True)

    tensorflow.config.threading.set_intra_op_parallelism_threads(num_cores)
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_cores)

    if len(gpu_devices) > 1:
        print(visible_devices)
        print(gpu_devices)
        strategy = tensorflow.distribute.MirroredStrategy(
            devices=[d.name[len('/physical_device:'):] for d in gpu_devices])
        #   cross_device_ops=tensorflow.contrib.distribute.AllReduceCrossDeviceOps(
        #      all_reduce_alg="hierarchical_copy")
    else:
        strategy = tensorflow.distribute.get_strategy()
    return strategy


# Example:
# python -u -m dmp.jobqueue_interface.worker dmp 10 '0,1' '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35' '0,1' 15360
if __name__ == "__main__":
    a = sys.argv
    print(a)
    
    

    database = a[1]
    queue_id = int(a[2])

    nodes = [int(e) for e in a[3].split(',')]
    cpus = [int(e) for e in a[4].split(',')]
    gpus = [int(e) for e in a[5].split(',')]
    gpu_memory = int(a[6])

    tensorflow.keras.backend.set_floatx('float32')

    worker_id = uuid.uuid4()
    print(f'Worker id {worker_id} starting...')
    print('\n', flush=True)

    if not isinstance(queue_id, int):
        queue_id = 1

    print(f'Worker id {worker_id} load credentials...\n', flush=True)
    credentials = load_credentials(database)
    print(f'Worker id {worker_id} initialize database schema...\n', flush=True)
    schema = PostgresSchema(credentials)
    print(f'Worker id {worker_id} create job queue...\n', flush=True)
    job_queue = JobQueue(credentials, int(queue_id), check_table=False)
    print(f'Worker id {worker_id} create result logger..\n', flush=True)
    result_logger = PostgresCompressedResultLogger(schema)
    print(f'Worker id {worker_id} create Worker object..\n', flush=True)

    strategy = make_strategy(
        len(cpus),
        gpus,
        gpu_memory,
    )

    worker = Worker(
        job_queue,
        schema,
        result_logger,
        strategy,
        {
            'nodes': nodes,
            'cpus': cpus,
            'gpus': gpus,
            'num_cpus': len(cpus),
            'num_nodes': len(nodes),
            'gpu_memory': gpu_memory,
            'tensorflow_strategy': str(type(strategy)),
            'queue_id': queue_id,
        },
    )
    print(f'Worker id {worker_id} start Worker object...\n', flush=True)
    worker()  # runs the work loop on the worker
    print(f'Worker id {worker_id} Worker exited.\n', flush=True)
