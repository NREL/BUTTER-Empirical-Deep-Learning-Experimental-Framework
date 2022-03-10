import argparse
import gc
import uuid


import jobqueue
import tensorflow

from .common import jobqueue_marshal


def make_strategy(cpu_low, cpu_high, gpu_low, gpu_high, gpu_mem):
    # tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    num_cpu = cpu_high - cpu_low
    num_gpu = gpu_high - gpu_low

    devices = []
    devices.extend(['/GPU:' + str(i) for i in range(gpu_low, gpu_high)])
    if num_cpu > num_gpu * 2:  # no CPU device if 2 or fewer CPUs per GPU
        devices.append('/CPU:0')  # TF batches all CPU's into one device

    # make sure we have one thread even if not using any CPUs
    num_threads = max(1, num_cpu)

    # gpus = tensorflow.config.experimental.list_physical_devices('GPU')  # Get GPU list
    # tensorflow.config.experimental.set_memory_growth(True)

    # print(tensorflow.config.experimental.list_physical_devices('GPU'))

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    # print(f'gpus: {gpus}')
    visible_devices = []
    for i in range(gpu_low, gpu_high):
        gpu = gpus[i]
        visible_devices.append(gpu)
        tensorflow.config.experimental.set_virtual_device_configuration(
            gpu,
            [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])

    cpus = tensorflow.config.experimental.list_physical_devices('CPU')
    # print(f'cpus: {cpus}')
    visible_devices.extend(cpus)
    tensorflow.config.set_visible_devices(visible_devices)

    tensorflow.config.threading.set_intra_op_parallelism_threads(num_threads)
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_threads)
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


def handle_job(worker_id: uuid.UUID, job: jobqueue.Job):
    gc.collect()
    task = jobqueue_marshal.demarshal(job.command)
    task(job.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cpu_low', type=int, help='minimum CPU id to use')
    parser.add_argument('cpu_high', type=int, help='1 + maximum CPU id to use')
    parser.add_argument('gpu_low', type=int, help='minimum GPU id to use')
    parser.add_argument('gpu_high', type=int, help='1 + maximum GPU id to use')
    parser.add_argument('gpu_mem', type=int, help='Per-GPU RAM to allocate')
    parser.add_argument(
        'database', help='database/project identifier in your jobqueue.json file')
    parser.add_argument('queue', help='queue id to use (smallint)')
    args = parser.parse_args()

    strategy = make_strategy(args.cpu_low, args.cpu_high,
                             args.gpu_low, args.gpu_high, args.gpu_mem)

    worker_id = uuid.uuid4()
    print(f'Worker id {worker_id} starting...')
    credentials = jobqueue.connect.load_credentials(args.database)
    job_queue = jobqueue.JobQueue(credentials, queue=args.queue)
    job_queue.run_worker(handle_job, worker_id=worker_id)
