import argparse
import gc
import json
import math
import sys
import uuid

import jobqueue.connect as connect
from jobqueue.job_queue import JobQueue
from dmp.logging.postgres_result_logger import PostgresResultLogger
from dmp.worker import Worker


import jobqueue
import tensorflow

from .common import jobqueue_marshal


def make_strategy(num_cores, first_gpu, num_gpus, gpu_mem):

    devices = []
    devices.extend(['/GPU:' + str(i)
                   for i in range(first_gpu, first_gpu + num_gpus)])
    # if num_core > num_gpu * 2:  # no CPU device if 2 or fewer CPUs per GPU
    devices.append('/CPU:0')  # TF batches all CPU's into one device

    # make sure we have one thread even if not using any CPUs
    num_threads = max(1, num_cores)

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    print(
        f'Found: {len(gpus)} GPUs. Using: {first_gpu} - {first_gpu + num_gpus}.')
    visible_devices = []
    for i in range(first_gpu, first_gpu + num_gpus):
        gpu = gpus[i]
        visible_devices.append(gpu)
        tensorflow.config.experimental.set_virtual_device_configuration(
            gpu,
            [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])

    cpus = tensorflow.config.experimental.list_physical_devices('CPU')
    # print(f'cpus: {cpus}')
    visible_devices.extend(cpus)
    tensorflow.config.set_visible_devices(visible_devices)
    tensorflow.config.set_soft_device_placement(True)

    tensorflow.config.threading.set_intra_op_parallelism_threads(num_cores)
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_cores)

    if num_gpus > 1:
        print(visible_devices)
        strategy = tensorflow.distribute.MirroredStrategy(
            devices=[d.name for d in visible_devices])
        #   cross_device_ops=tensorflow.contrib.distribute.AllReduceCrossDeviceOps(
        #      all_reduce_alg="hierarchical_copy")
    else:
        strategy = tensorflow.distribute.get_strategy()
    return strategy


if __name__ == "__main__":
    a = sys.argv

    num_cores = int(a[4])
    first_gpu = int(a[5])
    num_gpus = int(a[6])
    gpu_memory = int(a[7])

    database = a[8]
    queue = int(a[9])

    nodes = [int(e) for e in a[10].split(',')]
    cpus = [int(e) for e in a[11].split(',')]

    gpus = list(range(first_gpu, first_gpu + num_gpus))

    strategy = make_strategy(
        num_cores,
        first_gpu,
        num_gpus,
        gpu_memory,
    )

    worker_id = uuid.uuid4()
    print(f'Worker id {worker_id} starting...')
    print('\n', flush=True)

    # queue = args.queue

    if not isinstance(queue, int):
        queue = 1

    print(f'Worker id {worker_id} load credentials...\n', flush=True)
    credentials = connect.load_credentials(database)
    print(f'Worker id {worker_id} create job queue...\n', flush=True)
    job_queue = JobQueue(credentials, int(queue), check_table=False)
    print(f'Worker id {worker_id} create result logger..\n', flush=True)
    result_logger = PostgresResultLogger(credentials)
    print(f'Worker id {worker_id} create Worker object..\n', flush=True)
    with strategy.scope():
        worker = Worker(
            job_queue,
            result_logger,
            {
                'nodes': nodes,
                'cpus': cpus,
                'gpus': gpus,
                'num_cpus': len(cpus),
                'num_nodes': len(nodes),
                'gpu_memory': gpu_memory,
                'strategy': str(type(strategy)),
            },
        )
    print(f'Worker id {worker_id} start Worker object...\n', flush=True)
    worker()  # runs the work loop on the worker
    print(f'Worker id {worker_id} Worker exited.\n', flush=True)
