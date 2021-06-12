import gc

import dmp.experiment.aspect_test as exp
from dmp.data.logging import write_log
import tensorflow
import time
import jobqueue
import sys
import os


def start_jobqueue(strategy):
    jq = jobqueue.JobQueue(sys.argv[1], sys.argv[2])

    while True:

        # Pull job off the queue
        message = jq.get_message()

        if message is None:
            # No jobs, wait one second and try again.
            print("No Jobs, waiting")
            time.sleep(1)
            continue
        try:
            print(f"Job Queue: {message.uuid} RUNNING")

            # Run the experiment
            result = exp.aspect_test(message.config, strategy=strategy)

            # Write the log
            write_log(result, message.config["log"],
                      name=result["run_name"],
                      job=message.uuid)

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
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return tensorflow.Session(config=config)


## Set number of threads
# session_conf = tensorflow.ConfigProto(
#       intra_op_parallelism_threads=1,
#       inter_op_parallelism_threads=1)
# sess = tensorflow.Session(config=session_conf)

## Detect when more than one GPU with nvidia-smi

def make_strategy(DMP_CPU_LOW, DMP_CPU_HIGH, DMP_GPU_LOW, DMP_GPU_HIGH):
    # tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    devices = []
    devices.extend(['/CPU:' + str(i) for i in range(DMP_CPU_LOW, DMP_CPU_HIGH)])
    devices.extend(['/GPU:' + str(i) for i in range(DMP_GPU_LOW, DMP_GPU_HIGH)])


    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1)
    # sess = tf.Session(config=session_conf)

    num_cpu = DMP_CPU_HIGH - DMP_CPU_LOW
    tensorflow.config.threading.set_intra_op_parallelism_threads(num_cpu)
    tensorflow.config.threading.set_inter_op_parallelism_threads(num_cpu)

    strategy = tensorflow.distribute.MirroredStrategy(devices)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    return strategy
    #
    #
    # if type == "GPU":
    #     with tensorflow.device("/device:GPU:" + DMP_RANK):
    #         with tf_gpu_session().as_default():
    #             start_jobqueue()
    #
    # else:
    #     with tensorflow.device("/device:CPU:" + DMP_RANK):
    #         start_jobqueue()


# def start_jobqueue_with_device_based_on_type(DMP_TYPE, DMP_RANK):

#     devices = tensorflow.config.list_logical_devices(device_type=DMP_TYPE)

#     device = devices[DMP_RANK%len(devices)]

#     print(f"Starting {DMP_TYPE} rank {DMP_RANK} job queue on device {device.name}")

#     with tensorflow.device(device.name):

#         if type=="GPU":
#             with tf_gpu_session().as_default():
#                 start_jobqueue()

#         else:
#             start_jobqueue()


if __name__ == "__main__":
    # DMP_WORKER_ID = os.getenv('DMP_WORKER_ID')
    DMP_CPU_LOW = int(os.getenv('DMP_CPU_LOW'))
    DMP_CPU_HIGH = int(os.getenv('DMP_CPU_HIGH'))
    DMP_GPU_LOW = int(os.getenv('DMP_GPU_LOW'))
    DMP_GPU_HIGH = int(os.getenv('DMP_GPU_HIGH'))

    strategy = make_strategy(DMP_CPU_LOW, DMP_CPU_HIGH, DMP_GPU_LOW, DMP_GPU_HIGH)
    start_jobqueue(strategy)
