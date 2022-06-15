import argparse
import gc
import math
import uuid

import jobqueue.connect as connect
from jobqueue.job_queue import JobQueue
from dmp.logging.postgres_result_logger import PostgresResultLogger
from dmp.worker import Worker


import jobqueue
import tensorflow

from .common import jobqueue_marshal

def make_strategy(first_socket, num_sockets, first_core, num_cores, first_gpu, num_gpus, gpu_mem):
    
    devices = []
    devices.extend(['/GPU:' + str(i)
                   for i in range(first_gpu, first_gpu + num_gpus)])
    # if num_core > num_gpu * 2:  # no CPU device if 2 or fewer CPUs per GPU
    devices.append('/CPU:0')  # TF batches all CPU's into one device

    # make sure we have one thread even if not using any CPUs
    num_threads = max(1, num_cores)

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    print(f'Found: {len(gpus)} using: {first_gpu} - {first_gpu + num_gpus}.')
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

    tensorflow.config.threading.set_intra_op_parallelism_threads(
        int(math.ceil(num_cores / max(1, num_sockets))))
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_sockets)

    strategy = tensorflow.distribute.get_strategy()  # the default strategy
    return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('first_socket', type=int,
                        help='first socket id to use')
    parser.add_argument('num_sockets', type=int, help='num sockets to use')
    parser.add_argument('first_core', type=int, help='first core id to use')
    parser.add_argument('num_cores', type=int, help='num cores to use')
    parser.add_argument('first_gpu', type=int, help='first GPU id to use')
    parser.add_argument('num_gpus', type=int, help='num GPUs to use')
    parser.add_argument('gpu_mem', type=int, help='GPU RAM to allocate')
    parser.add_argument(
        'database', help='database/project identifier in your jobqueue.json file')
    parser.add_argument('queue', help='queue id to use (smallint)')
    args = parser.parse_args()

    strategy = make_strategy(
        args.first_socket, args.num_sockets,
        args.first_core, args.num_cores,
        args.first_gpu, args.num_gpus,
        args.gpu_mem)

    worker_id = uuid.uuid4()
    print(f'Worker id {worker_id} starting...')
    print('\n', flush=True)

    queue = args.queue
    if not isinstance(queue, int):
        queue = 1

    print(f'Worker id {worker_id} load credentials...\n', flush=True)
    credentials = connect.load_credentials('dmp')
    print(f'Worker id {worker_id} create job queue...\n', flush=True)
    job_queue = JobQueue(credentials, int(queue), check_table=False)
    print(f'Worker id {worker_id} create result logger..\n', flush=True)
    result_logger = PostgresResultLogger(credentials)
    print(f'Worker id {worker_id} create Worker object..\n', flush=True)
    worker = Worker(job_queue, result_logger)
    print(f'Worker id {worker_id} start Worker object...\n', flush=True)
    worker()  # runs the work loop on the worker
    print(f'Worker id {worker_id} Worker exited.\n', flush=True)
