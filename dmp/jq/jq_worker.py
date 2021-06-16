import argparse
import gc
import random

import dmp.experiment.aspect_test as exp
from dmp.data.logging import write_log
import tensorflow
import time
import jobqueue
import sys
import os


def run_worker(strategy, config, project, group, max_waiting_time=10 * 60):
    print(f"Job Queue: Starting...")

    # with tensorflow.Session(config=config):
    jq = jobqueue.JobQueue(project, group)
    wait_start = None
    while True:

        # Pull job off the queue
        message = jq.get_message()

        if message is None:

            if wait_start is None:
                wait_start = time.time()
            else:
                waiting_time = time.time() - wait_start
                if waiting_time > max_waiting_time:
                    print("Job Queue: No Jobs, max waiting time exceeded. Exiting...")
                    break

            # No jobs, wait one second and try again.
            print("Job Queue: No jobs found. Waiting...")
            time.sleep(random.randint(1, 10))  # TODO: could use exponential backoff...
            continue
        try:
            wait_start = None
            print(f"Job Queue: {message.uuid} running")

            # Run the experiment
            result = exp.aspect_test(message.config, strategy=strategy)

            # Write the log
            write_log(result,
                      message.config["log"],
                      name=result["run_name"],
                      job=message.uuid,
                      )

            # Mark the job as complete in the queue.
            message.mark_complete()
            gc.collect()

            print(f"Job Queue: {message.uuid} DONE")
        except Exception as e:
            print(f"Job Queue: {message.uuid} Unknown exception in jq_runner: {e}")


# def tf_gpu_session():
#     """
#     Return a tensorflow session with the configuration we'll use on a GPU node.
#     """
#     config = tensorflow.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = .5
#     return tensorflow.Session(config=config)


def make_strategy(cpu_low, cpu_high, gpu_low, gpu_high, gpu_mem):
    # tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    num_cpu = cpu_high - cpu_low
    num_gpu = gpu_high - gpu_low



    devices = []
    devices.extend(['/GPU:' + str(i) for i in range(gpu_low, gpu_high)])
    if num_cpu > num_gpu * 2:  # no CPU device if 2 or fewer CPUs per GPU
        devices.append('/CPU:0')  # TF batches all CPU's into one device

    num_threads = max(1, num_cpu)  # make sure we have one thread even if not using any CPUs

    # gpus = tensorflow.config.experimental.list_physical_devices('GPU')  # Get GPU list
    # tensorflow.config.experimental.set_memory_growth(True)

    # print(tensorflow.config.experimental.list_physical_devices('GPU'))

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    print(f'gpus: {gpus}')
    visible_devices = []
    for i in range(gpu_low, gpu_high):
        gpu = gpus[i]
        visible_devices.append(gpu)
        tensorflow.config.experimental.set_virtual_device_configuration(
            gpu,
            [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])

    cpus = tensorflow.config.experimental.list_physical_devices('CPU')
    print(f'cpus: {cpus}')
    visible_devices.extend(cpus)
    tensorflow.config.set_visible_devices(visible_devices)

    # for gpu in gpus:
    #     tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
    #         tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    # tensorflow.config.threading.set_intra_op_parallelism_threads(num_threads)
    # tensorflow.config.threading.set_inter_op_parallelism_threads(num_threads)
    # tensorflow.config.set_visible_devices(devices)

    # gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.995)
    # cpu_options = tensorflow.CPU
    # config = tensorflow.ConfigProto(gpu_options=gpu_options)

    print(f'num_cpu {num_cpu}, num_gpu {num_gpu}, num_threads {num_threads}')

    if len(devices) == 1:
        # if num_gpu > 1:
        #     strategy = tensorflow.distribute.OneDeviceStrategy(device=devices[0])
        # else:
        strategy = tensorflow.distribute.get_strategy()  # the default strategy
    else:
        strategy = tensorflow.distribute.MirroredStrategy(devices=devices)

    print('num_replicas_in_sync: {}'.format(strategy.num_replicas_in_sync))
    return strategy, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cpu_low', type=int, help='minimum CPU id to use')
    parser.add_argument('cpu_high', type=int, help='1 + maximum CPU id to use')
    parser.add_argument('gpu_low', type=int, help='minimum GPU id to use')
    parser.add_argument('gpu_high', type=int, help='1 + maximum GPU id to use')
    parser.add_argument('gpu_mem', type=int, help='Per-GPU RAM to allocate')
    parser.add_argument('project', help='project identifier in your jobqueue.json file')
    parser.add_argument('group', help='group name or tag')
    args = parser.parse_args()

    strategy, config = make_strategy(args.cpu_low, args.cpu_high, args.gpu_low, args.gpu_high, args.gpu_mem)
    run_worker(strategy, config, args.project, args.group)
