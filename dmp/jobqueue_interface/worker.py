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

"""
TODO:
python -m dmp.jq.jq_node_manager dmp 1 "[0, [0, 6, 0, 0, 0], [6, 12, 0, 0, 0], [12, 18, 0, 0, 0], [18, 24, 0, 0, 0], [24, 30, 0, 0, 0]]"

 + jq_node_manager:
    + identify system configuration
        + num gpus, gpu memory, num cpus
    + call node manager with appropriate args:
        + cpu call 

 + Create python program or script to get number of gpus and return it in stdout
 + Replace worker script with:
    + Check GPUs, Memory: https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
        + ~one GPU worker per 6 GB? 1 or 2 threads for this worker.
    + Remaining threads run in cpu environment on single worker.

    total_cpu_cores=$(nproc)
    number_sockets=$(($(grep "^physical id" /proc/cpuinfo | awk '{print $4}' | sort -un | tail -1)+1))
    number_cpu_cores=$(( (total_cpu_cores/2) / number_sockets))

    export KMP_BLOCKTIME=1
    export OMP_NUM_THREADS= #physical cores
    export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

    intra_op_parallelism = number of physical core per socket
    inter_op_parallelism = number of sockets
"""


def make_strategy(first_socket, num_sockets, first_core, num_cores, first_gpu, num_gpus, gpu_mem):
    # tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    devices = []
    devices.extend(['/GPU:' + str(i)
                   for i in range(first_gpu, first_gpu + num_gpus)])
    # if num_core > num_gpu * 2:  # no CPU device if 2 or fewer CPUs per GPU
    devices.append('/CPU:0')  # TF batches all CPU's into one device

    # make sure we have one thread even if not using any CPUs
    num_threads = max(1, num_cores)

    # gpus = tensorflow.config.experimental.list_physical_devices('GPU')  # Get GPU list
    # tensorflow.config.experimental.set_memory_growth(True)

    # print(tensorflow.config.experimental.list_physical_devices('GPU'))

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
        int(math.ceil(num_cores / num_sockets)))
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_sockets)
    # tensorflow.config.set_visible_devices(devices)

    # gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.995)
    # cpu_options = tensorflow.CPU
    # config = tensorflow.ConfigProto(gpu_options=gpu_options)

    # print(f'num_cpu {num_cpu}, num_gpu {num_gpu}, num_threads {num_threads}')

    # if len(devices) == 1:
    #     # if num_gpu > 1:
    #     strategy = tensorflow.distribute.OneDeviceStrategy(device=devices[0])
    #     # else:
    #     #     strategy = tensorflow.distribute.get_strategy()  # the default strategy
    # else:
    # strategy = tensorflow.distribute.MirroredStrategy(devices=devices)
    strategy = tensorflow.distribute.get_strategy()  # the default strategy

    # print('num_replicas_in_sync: {}'.format(strategy.num_replicas_in_sync))
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
