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


def run_worker(strategy, project, group, max_waiting_time=10 * 60):
    print(f"Job Queue: Starting...")

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


def tf_gpu_session():
    """
    Return a tensorflow session with the configuration we'll use on a GPU node.
    """
    config = tensorflow.ConfigProto()
    # config.gpu_options.allow_growth = False
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return tensorflow.Session(config=config)


def make_strategy(cpu_min, cpu_max, gpu_min, gpu_max):
    # tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    num_cpu = cpu_max - cpu_min
    num_gpu = gpu_max - gpu_min

    devices = []
    devices.extend(['/GPU:' + str(i) for i in range(gpu_min, gpu_max)])
    if num_cpu > num_gpu * 2:  # no CPU device if 2 or fewer CPUs per GPU
        devices.append('/CPU:0')  # TF batches all CPU's into one device

    num_threads = max(1, num_cpu)  # make sure we have one thread even if not using any CPUs
    tensorflow.config.threading.set_intra_op_parallelism_threads(num_threads)
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_threads)

    if len(devices) == 1:
        strategy = tensorflow.distribute.OneDeviceStrategy(device=devices[0])
    else:
        strategy = tensorflow.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=tensorflow.contrib.distribute.AllReduceCrossDeviceOps(all_reduce_alg="hierarchical_copy"))

    print('num_replicas_in_sync: {}'.format(strategy.num_replicas_in_sync))
    return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cpu_low', type=int, help='minimum CPU id to use')
    parser.add_argument('cpu_high', type=int, help='1 + maximum CPU id to use')
    parser.add_argument('gpu_low', type=int, help='minimum GPU id to use')
    parser.add_argument('gpu_high', type=int, help='1 + maximum GPU id to use')
    parser.add_argument('project', help='project identifier in your jobqueue.json file')
    parser.add_argument('group', help='group name or tag')
    args = parser.parse_args()

    strategy = make_strategy(args.cpu_low, args.gpu_high, args.gpu_low, args.gpu_high)
    run_worker(strategy, args.project, args.group)
